"""SQLite 持久化层：按群保存召集配置、聊天文本和用户模仿授权。"""

import sqlite3
import threading
import time
from pathlib import Path


# 数据表用途：managed_users 存名单，trigger_keywords 存关键词，
# messages 存群聊文本和样本资格，mimic_consent 存各用户在各群的授权。
# WAL 日志模式允许读写操作更好地共存；群号参与键值，隔离不同群的数据。
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS managed_users (
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    alias TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);
CREATE TABLE IF NOT EXISTS trigger_keywords (
    group_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, keyword)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    nickname TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    mimic_eligible INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_group_time
    ON messages(group_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON messages(group_id, user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS mimic_consent (
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    consented_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);
"""


class Database:
    """封装同步数据库操作，通过锁保护共享连接，并用事务提交写入。"""

    def __init__(self, path: Path):
        """创建数据库目录、连接、表结构，并兼容早期数据库缺失的字段。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        # log_message 持锁时会调用同样需要锁的授权查询，因此使用可重入锁。
        self._lock = threading.RLock()
        with self._lock:
            self.connection.executescript(SCHEMA)
            # 老版本消息表可能没有样本资格字段，补充时默认不授权历史消息。
            columns = {
                row["name"]
                for row in self.connection.execute("PRAGMA table_info(messages)")
            }
            if "mimic_eligible" not in columns:
                self.connection.execute(
                    "ALTER TABLE messages ADD COLUMN mimic_eligible INTEGER NOT NULL DEFAULT 0"
                )

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self.connection.close()

    def add_managed_user(self, group_id: int, user_id: int, alias: str = "") -> None:
        """添加名单成员；已存在时更新别名并恢复启用状态。"""
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO managed_users(group_id,user_id,alias,enabled,created_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(group_id,user_id) DO UPDATE SET alias=excluded.alias,enabled=1""",
                (group_id, user_id, alias, 1, int(time.time())),
            )

    def remove_managed_user(self, group_id: int, user_id: int) -> bool:
        """从指定群的名单移除成员，返回是否实际删除了记录。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM managed_users WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            )
            return cursor.rowcount > 0

    def list_managed_users(self, group_id: int) -> list[sqlite3.Row]:
        """按创建时间返回指定群中已启用成员的 QQ 号和别名。"""
        with self._lock:
            return list(
                self.connection.execute(
                    "SELECT user_id,alias FROM managed_users WHERE group_id=? AND enabled=1 ORDER BY created_at",
                    (group_id,),
                )
            )

    def add_keyword(self, group_id: int, keyword: str) -> None:
        """添加当前群的触发关键词，重复添加同一关键词不会产生新记录。"""
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO trigger_keywords(group_id,keyword,created_at) VALUES(?,?,?)",
                (group_id, keyword, int(time.time())),
            )

    def remove_keyword(self, group_id: int, keyword: str) -> bool:
        """删除当前群的指定关键词，返回是否找到并删除记录。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM trigger_keywords WHERE group_id=? AND keyword=?",
                (group_id, keyword),
            )
            return cursor.rowcount > 0

    def list_keywords(self, group_id: int) -> list[str]:
        """按创建时间返回当前群的全部关键词。"""
        with self._lock:
            return [
                row["keyword"]
                for row in self.connection.execute(
                    "SELECT keyword FROM trigger_keywords WHERE group_id=? ORDER BY created_at",
                    (group_id,),
                )
            ]

    def log_message(
        self, group_id: int, user_id: int, nickname: str, content: str
    ) -> None:
        """保存非空文本，最多 4000 字；按保存时的授权状态标记样本资格。"""
        if not content:
            return
        with self._lock, self.connection:
            # 将当时的授权状态存入消息，之后授权不会让之前的发言自动成为样本。
            mimic_eligible = int(self.has_mimic_consent(group_id, user_id))
            self.connection.execute(
                """INSERT INTO messages(group_id,user_id,nickname,content,mimic_eligible,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (group_id, user_id, nickname, content[:4000], mimic_eligible, int(time.time())),
            )

    def recent_messages(self, group_id: int, limit: int) -> list[sqlite3.Row]:
        """取当前群最新的 limit 条消息，再按从旧到新的顺序返回给模型。"""
        with self._lock:
            rows = list(
                self.connection.execute(
                    """SELECT user_id,nickname,content FROM messages
                       WHERE group_id=? ORDER BY id DESC LIMIT ?""",
                    (group_id, limit),
                )
            )
        # 倒序查询便于限制最新条数，反转后符合聊天上下文的时间顺序。
        rows.reverse()
        return rows

    def set_mimic_consent(self, group_id: int, user_id: int, enabled: bool) -> None:
        """设置用户在当前群的授权；撤销只删除授权记录，不删除历史消息。"""
        with self._lock, self.connection:
            if enabled:
                self.connection.execute(
                    "INSERT OR REPLACE INTO mimic_consent(group_id,user_id,consented_at) VALUES(?,?,?)",
                    (group_id, user_id, int(time.time())),
                )
            else:
                self.connection.execute(
                    "DELETE FROM mimic_consent WHERE group_id=? AND user_id=?",
                    (group_id, user_id),
                )

    def has_mimic_consent(self, group_id: int, user_id: int) -> bool:
        """查询用户是否仍允许在当前群被模仿。"""
        with self._lock:
            return self.connection.execute(
                "SELECT 1 FROM mimic_consent WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            ).fetchone() is not None

    def style_samples(self, group_id: int, user_id: int, limit: int) -> list[str]:
        """取目标用户授权期间保存的最新文本，按从旧到新返回。

        此方法只检查消息的样本资格，调用方还需要确认用户当前仍然授权。
        """
        with self._lock:
            rows = list(
                self.connection.execute(
                    """SELECT content FROM messages
                       WHERE group_id=? AND user_id=? AND mimic_eligible=1
                       ORDER BY id DESC LIMIT ?""",
                    (group_id, user_id, limit),
                )
            )
        return [row["content"] for row in reversed(rows)]

    def delete_user_messages(self, group_id: int, user_id: int) -> int:
        """删除用户在指定群的全部消息并撤销模仿授权，返回删除的消息条数。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM messages WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            )
            self.connection.execute(
                "DELETE FROM mimic_consent WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            )
            return cursor.rowcount

    def purge_old_messages(self, retention_days: int) -> int:
        """删除超过保留天数的消息；非正数表示不清理，返回删除条数。"""
        if retention_days <= 0:
            return 0
        cutoff = int(time.time()) - retention_days * 86400
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM messages WHERE created_at < ?", (cutoff,)
            )
            return cursor.rowcount

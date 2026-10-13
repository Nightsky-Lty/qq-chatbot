"""SQLite 持久化层：保存群配置及聊天，私聊记录独立按用户隔离。"""

import json
import sqlite3
import threading
import time
from pathlib import Path


# 数据表用途：member_lists 存名单，list_members 存成员，trigger_keywords 存关键词，
# keyword_list_bindings 保存关键词到名单的唯一绑定；外键防止悬空引用。
# messages 存群聊文本，普通文字发言可用作当前群的模仿样本。
# private_messages 只存每位用户与机器人的私聊，不作为群聊上下文或模仿样本。
# WAL 日志模式允许读写操作更好地共存；群号参与键值，隔离不同群的数据。
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS member_lists (
    group_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, name)
);
CREATE TABLE IF NOT EXISTS list_members (
    group_id INTEGER NOT NULL,
    list_name TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    alias TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, list_name, user_id),
    FOREIGN KEY (group_id, list_name) REFERENCES member_lists(group_id, name)
        ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS trigger_keywords (
    group_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, keyword)
);
CREATE TABLE IF NOT EXISTS keyword_list_bindings (
    group_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    list_name TEXT NOT NULL,
    PRIMARY KEY (group_id, keyword),
    FOREIGN KEY (group_id, keyword) REFERENCES trigger_keywords(group_id, keyword)
        ON DELETE CASCADE,
    FOREIGN KEY (group_id, list_name) REFERENCES member_lists(group_id, name)
        ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    nickname TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_group_time
    ON messages(group_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON messages(group_id, user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS private_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_private_messages_user_id
    ON private_messages(user_id, id DESC);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    due_at INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','sent','cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status,due_at,next_attempt_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reminders_pending_unique
    ON reminders(user_id,content,due_at) WHERE status='pending';
CREATE TABLE IF NOT EXISTS sticker_annotations (
    content_hash TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('ready','quarantined','error')),
    metadata TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);
"""


class Database:
    """封装同步数据库操作，通过锁保护共享连接，并用事务提交写入。"""

    def __init__(self, path: Path):
        """创建数据库目录、连接和表结构，并迁移早期的单名单配置。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        # 绑定关键词时会嵌套调用持锁的名单查询，因此使用可重入锁。
        self._lock = threading.RLock()
        with self._lock:
            self.connection.executescript(SCHEMA)
        with self._lock, self.connection:
            # 旧库的授权表和样本资格字段不再读写，原样保留，避免为升级重建消息表。
            self._migrate_legacy_lists()

    def _migrate_legacy_lists(self) -> None:
        """在初始化事务中迁移旧单名单；全部复制成功后才移除旧表，避免重复导入。"""
        legacy = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='managed_users'"
        ).fetchone()
        if legacy is None:
            return
        # 有旧成员或旧关键词的群统一建立“默认”名单，保留原来的触发关系。
        self.connection.execute(
            """INSERT OR IGNORE INTO member_lists(group_id,name,created_at)
               SELECT group_id,'默认',? FROM (
                   SELECT group_id FROM managed_users
                   UNION SELECT group_id FROM trigger_keywords
               )""",
            (int(time.time()),),
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO list_members
               (group_id,list_name,user_id,alias,enabled,created_at)
               SELECT group_id,'默认',user_id,alias,enabled,created_at FROM managed_users"""
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO keyword_list_bindings(group_id,keyword,list_name)
               SELECT group_id,keyword,'默认' FROM trigger_keywords"""
        )
        self.connection.execute("DROP TABLE managed_users")

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self.connection.close()

    def get_sticker_annotation(self, content_hash: str) -> dict | None:
        """按内容哈希复用识别结果；不保存图片字节或私聊数据。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM sticker_annotations WHERE content_hash=?", (content_hash,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def save_sticker_annotation(
        self, content_hash: str, status: str, metadata: dict, *, attempts: int = 0,
        next_attempt_at: int = 0,
    ) -> None:
        """缓存安全判断或带退避时间的失败，重启后不会重复识别成功图片。"""
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO sticker_annotations
                   (content_hash,status,metadata,attempts,next_attempt_at,updated_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(content_hash) DO UPDATE SET
                   status=excluded.status,metadata=excluded.metadata,attempts=excluded.attempts,
                   next_attempt_at=excluded.next_attempt_at,updated_at=excluded.updated_at""",
                (content_hash, status, json.dumps(metadata, ensure_ascii=False), attempts,
                 next_attempt_at, int(time.time())),
            )

    def create_list(self, group_id: int, name: str) -> bool:
        """创建独立的命名名单，返回是否新建；不同群可使用相同名单名。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO member_lists(group_id,name,created_at) VALUES(?,?,?)",
                (group_id, name, int(time.time())),
            )
            return cursor.rowcount > 0

    def has_list(self, group_id: int, name: str) -> bool:
        """查询当前群是否存在指定名单。"""
        with self._lock:
            return self.connection.execute(
                "SELECT 1 FROM member_lists WHERE group_id=? AND name=?", (group_id, name)
            ).fetchone() is not None

    def list_lists(self, group_id: int) -> list[sqlite3.Row]:
        """列出当前群的所有名单及已启用成员数，包括空名单。"""
        with self._lock:
            return list(self.connection.execute(
                """SELECT l.name,COUNT(m.user_id) AS member_count FROM member_lists l
                   LEFT JOIN list_members m ON m.group_id=l.group_id
                       AND m.list_name=l.name AND m.enabled=1
                   WHERE l.group_id=? GROUP BY l.name ORDER BY l.created_at,l.name""",
                (group_id,),
            ))

    def delete_list(self, group_id: int, name: str) -> bool:
        """删除名单并级联删除成员和绑定，但保留独立的关键词。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM member_lists WHERE group_id=? AND name=?", (group_id, name)
            )
            return cursor.rowcount > 0

    def add_managed_user(self, group_id: int, list_name: str, user_id: int, alias: str = "") -> None:
        """添加名单成员；已存在时更新别名并恢复启用状态。"""
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO list_members(group_id,list_name,user_id,alias,enabled,created_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(group_id,list_name,user_id) DO UPDATE SET alias=excluded.alias,enabled=1""",
                (group_id, list_name, user_id, alias, 1, int(time.time())),
            )

    def remove_managed_user(self, group_id: int, list_name: str, user_id: int) -> bool:
        """从指定群的名单移除成员，返回是否实际删除了记录。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM list_members WHERE group_id=? AND list_name=? AND user_id=?",
                (group_id, list_name, user_id),
            )
            return cursor.rowcount > 0

    def list_managed_users(self, group_id: int, list_name: str) -> list[sqlite3.Row]:
        """按创建时间返回指定群、指定名单中已启用成员的 QQ 号和别名。"""
        with self._lock:
            return list(
                self.connection.execute(
                    """SELECT user_id,alias FROM list_members
                       WHERE group_id=? AND list_name=? AND enabled=1 ORDER BY created_at,user_id""",
                    (group_id, list_name),
                )
            )

    def add_keyword(self, group_id: int, keyword: str) -> bool:
        """添加当前群的触发关键词，重复添加同一关键词不会产生新记录。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO trigger_keywords(group_id,keyword,created_at) VALUES(?,?,?)",
                (group_id, keyword, int(time.time())),
            )
            return cursor.rowcount > 0

    def remove_keyword(self, group_id: int, keyword: str) -> bool:
        """删除当前群的指定关键词，返回是否找到并删除记录。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM trigger_keywords WHERE group_id=? AND keyword=?",
                (group_id, keyword),
            )
            return cursor.rowcount > 0

    def list_keywords(self, group_id: int) -> list[sqlite3.Row]:
        """列出关键词和绑定名单；未绑定时 list_name 为 None。"""
        with self._lock:
            return list(self.connection.execute(
                """SELECT k.keyword,b.list_name FROM trigger_keywords k
                   LEFT JOIN keyword_list_bindings b
                       ON b.group_id=k.group_id AND b.keyword=k.keyword
                   WHERE k.group_id=? ORDER BY k.created_at,k.keyword""",
                (group_id,),
            ))

    def keyword_list(self, group_id: int, keyword: str) -> str | None:
        """精确查询关键词绑定的唯一名单，未绑定或关键词不存在时返回 None。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT list_name FROM keyword_list_bindings WHERE group_id=? AND keyword=?",
                (group_id, keyword),
            ).fetchone()
            return row["list_name"] if row else None

    def bind_keyword(self, group_id: int, keyword: str, list_name: str) -> str:
        """绑定已有关键词与名单；不覆盖已有绑定，主键保证一个关键词至多一个名单。"""
        with self._lock, self.connection:
            if self.connection.execute(
                "SELECT 1 FROM trigger_keywords WHERE group_id=? AND keyword=?", (group_id, keyword)
            ).fetchone() is None:
                return "missing_keyword"
            if not self.has_list(group_id, list_name):
                return "missing_list"
            current = self.keyword_list(group_id, keyword)
            if current is not None:
                return "already_bound" if current == list_name else "conflict"
            self.connection.execute(
                "INSERT INTO keyword_list_bindings(group_id,keyword,list_name) VALUES(?,?,?)",
                (group_id, keyword, list_name),
            )
            return "bound"

    def unbind_keyword(self, group_id: int, keyword: str) -> bool:
        """只删除关键词与名单的绑定，不删除关键词或名单成员。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM keyword_list_bindings WHERE group_id=? AND keyword=?", (group_id, keyword)
            )
            return cursor.rowcount > 0

    def log_message(
        self, group_id: int, user_id: int, nickname: str, content: str
    ) -> None:
        """保存非空群聊文本，最多 4000 字，不再检查或记录模仿授权。"""
        if not content:
            return
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO messages(group_id,user_id,nickname,content,created_at)
                   VALUES(?,?,?,?,?)""",
                (group_id, user_id, nickname, content[:4000], int(time.time())),
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

    def log_private_message(self, user_id: int, role: str, content: str) -> None:
        """保存私聊用户文本或已提交发送的机器人回复，每条最多 4000 字。"""
        if not content:
            return
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO private_messages(user_id,role,content,created_at) VALUES(?,?,?,?)",
                (user_id, role, content[:4000], int(time.time())),
            )

    def recent_private_messages(self, user_id: int, limit: int) -> list[sqlite3.Row]:
        """只读取该用户与机器人的私聊，按时间顺序返回角色和文本。"""
        with self._lock:
            rows = list(self.connection.execute(
                """SELECT role,content FROM private_messages
                   WHERE user_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ))
        rows.reverse()
        return rows

    def delete_private_messages(self, user_id: int) -> int:
        """删除该用户的全部私聊记录，不影响群聊或名单。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM private_messages WHERE user_id=?", (user_id,)
            )
            return cursor.rowcount

    def delete_private_data(self, user_id: int) -> tuple[int, int]:
        """原子删除本人的私聊及全部提醒记录，返回两类记录的数量。"""
        with self._lock, self.connection:
            messages = self.connection.execute("DELETE FROM private_messages WHERE user_id=?", (user_id,))
            reminders = self.connection.execute("DELETE FROM reminders WHERE user_id=?", (user_id,))
            return messages.rowcount, reminders.rowcount

    def create_reminder(self, user_id: int, content: str, due_at: int, max_pending: int) -> tuple[sqlite3.Row, bool]:
        """创建持久提醒；同人同内容同时间的未完成提醒复用，防止工具重复调用。"""
        with self._lock, self.connection:
            existing = self.connection.execute(
                "SELECT * FROM reminders WHERE user_id=? AND content=? AND due_at=? AND status='pending'",
                (user_id, content, due_at),
            ).fetchone()
            if existing is not None:
                return existing, False
            count = self.connection.execute(
                "SELECT COUNT(*) FROM reminders WHERE user_id=? AND status='pending'", (user_id,)
            ).fetchone()[0]
            if count >= max_pending:
                raise ValueError(f"未完成提醒已达到 {max_pending} 条，请先取消不需要的提醒。")
            now = int(time.time())
            cursor = self.connection.execute(
                "INSERT INTO reminders(user_id,content,due_at,created_at,updated_at) VALUES(?,?,?,?,?)",
                (user_id, content, due_at, now, now),
            )
            row = self.connection.execute("SELECT * FROM reminders WHERE id=?", (cursor.lastrowid,)).fetchone()
            return row, True

    def list_reminders(self, user_id: int) -> list[sqlite3.Row]:
        """只查看当前用户的未完成提醒，包括等待连接恢复的过期提醒。"""
        with self._lock:
            return list(self.connection.execute(
                "SELECT * FROM reminders WHERE user_id=? AND status='pending' ORDER BY due_at,id", (user_id,)
            ))

    def get_reminder(self, user_id: int, reminder_id: int) -> sqlite3.Row | None:
        """按所有者与编号查询，防止工具访问其他用户的提醒。"""
        with self._lock:
            return self.connection.execute(
                "SELECT * FROM reminders WHERE user_id=? AND id=?", (user_id, reminder_id)
            ).fetchone()

    def cancel_reminder(self, user_id: int, reminder_id: int) -> bool:
        """仅取消属于当前用户且尚未完成的提醒。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE reminders SET status='cancelled',updated_at=? WHERE user_id=? AND id=? AND status='pending'",
                (int(time.time()), user_id, reminder_id),
            )
            return cursor.rowcount > 0

    def due_reminders(self, now: int, limit: int = 20) -> list[sqlite3.Row]:
        """供单实例后台调度查询到期且满足重试时间的提醒。"""
        with self._lock:
            return list(self.connection.execute(
                """SELECT * FROM reminders WHERE status='pending' AND due_at<=? AND next_attempt_at<=?
                   ORDER BY due_at,id LIMIT ?""", (now, now, limit)
            ))

    def mark_reminder_sent(self, reminder_id: int, now: int) -> bool:
        """收到 OneBot 成功响应后标记完成，不覆盖并发取消或删除的状态。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE reminders SET status='sent',updated_at=? WHERE id=? AND status='pending'",
                (now, reminder_id),
            )
            return cursor.rowcount > 0

    def retry_reminder(self, reminder_id: int, next_attempt_at: int, now: int) -> None:
        """发送失败仍保留待办，并更新重试次数及时间，不恢复已取消的提醒。"""
        with self._lock, self.connection:
            self.connection.execute(
                """UPDATE reminders SET attempts=attempts+1,next_attempt_at=?,updated_at=?
                   WHERE id=? AND status='pending'""", (next_attempt_at, now, reminder_id)
            )

    def style_samples(self, group_id: int, user_id: int, limit: int) -> list[str]:
        """取目标用户在当前群的最新普通文字，排除空文本及 / 指令，不读私聊。"""
        with self._lock:
            rows = list(
                self.connection.execute(
                    """SELECT content FROM messages
                       WHERE group_id=? AND user_id=?
                           AND trim(content)<>'' AND ltrim(content) NOT LIKE '/%'
                       ORDER BY id DESC LIMIT ?""",
                    (group_id, user_id, limit),
                )
            )
        return [row["content"] for row in reversed(rows)]

    def delete_user_messages(self, group_id: int, user_id: int) -> int:
        """删除用户在指定群的全部消息及模仿样本，不影响名单或私聊。"""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM messages WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            )
            return cursor.rowcount

    def purge_old_messages(self, retention_days: int) -> int:
        """清理过期消息及已完成/取消的提醒；待办不清理，返回总删除条数。"""
        if retention_days <= 0:
            return 0
        cutoff = int(time.time()) - retention_days * 86400
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM messages WHERE created_at < ?", (cutoff,)
            )
            private_cursor = self.connection.execute(
                "DELETE FROM private_messages WHERE created_at < ?", (cutoff,)
            )
            reminders = self.connection.execute(
                "DELETE FROM reminders WHERE status<>'pending' AND updated_at<?", (cutoff,)
            )
            return cursor.rowcount + private_cursor.rowcount + reminders.rowcount

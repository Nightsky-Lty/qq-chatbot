"""SQLite 持久化层：保存群配置及聊天，私聊记录独立按用户隔离。"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path


# 数据表用途：member_lists 存名单，list_members 存成员，trigger_keywords 存关键词，
# keyword_list_bindings 保存关键词到名单的唯一绑定；外键防止悬空引用。
# messages 存群聊文本、机器人成功回复和动作通知；只有user普通文字可作本群模仿样本。
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
    message_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user',
    reply_to_message_id TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_group_time
    ON messages(group_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON messages(group_id, user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS group_members (
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    role TEXT NOT NULL DEFAULT 'member',
    first_seen_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);
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
CREATE TABLE IF NOT EXISTS memory_capture_state (
    namespace INTEGER PRIMARY KEY,
    cursor_id INTEGER NOT NULL,
    requested_id INTEGER NOT NULL,
    due_at INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0
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
        self.capture_enabled = False
        self.capture_delay = 30
        self.message_listener = None
        # 仅初始化当前分层记忆表，不读取旧记忆存储。
        from app.memory_schema import initialize_memory_schema
        try:
            initialize_memory_schema(self.connection)
        except BaseException:
            self.connection.close()
            raise
        with self._lock:
            self.connection.executescript(SCHEMA)
        with self._lock, self.connection:
            columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(messages)")}
            for name, default in (("message_id", ""), ("role", "user"), ("reply_to_message_id", "")):
                if name not in columns:
                    self.connection.execute(f"ALTER TABLE messages ADD COLUMN {name} TEXT NOT NULL DEFAULT '{default}'")
            if "mentions_json" not in columns:
                self.connection.execute("ALTER TABLE messages ADD COLUMN mentions_json TEXT NOT NULL DEFAULT '[]'")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_onebot ON messages(group_id,message_id)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_group_role_id ON messages(group_id,role,id DESC)")
            # 旧库的授权表和样本资格字段不再读写，原样保留，避免为升级重建消息表。
            self._migrate_legacy_lists()
        # 运行时只访问当前分层存储。
        from app.memory_store import MemoryStore
        self.memories = MemoryStore(self)

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

    def _queue_memory_capture(self, namespace: int, record_id: int) -> None:
        """与消息入库同一事务排队；正群号、负私聊QQ构成互斥命名空间。"""
        if not self.capture_enabled:
            return
        due = int(time.time()) + self.capture_delay
        self.connection.execute(
            """INSERT INTO memory_capture_state(namespace,cursor_id,requested_id,due_at)
               VALUES(?,?,?,?) ON CONFLICT(namespace) DO UPDATE SET
               requested_id=max(requested_id,excluded.requested_id),
               due_at=CASE WHEN cursor_id>=requested_id THEN excluded.due_at ELSE due_at END""",
            (namespace, record_id - 1, record_id, due),
        )

    def _assign_memory_reply(self, namespace: int, record_id: int, reply_to: str = ""):
        """实际发送的回复在同一事务归属任务，防止后台把未结束任务拆成两次新证据。"""
        if not self.capture_enabled:
            return
        if namespace > 0:
            task = self.connection.execute("""SELECT j.task_id FROM memory_consolidation_jobs j
              JOIN memory_task_sources s ON s.task_id=j.task_id AND s.namespace=j.namespace
              JOIN messages m ON m.id=s.source_id AND m.group_id=j.namespace
              WHERE j.namespace=? AND j.status='running' AND m.message_id=? ORDER BY j.created_at DESC LIMIT 1""", (namespace, reply_to)).fetchone() if reply_to else None
        else:
            task = self.connection.execute("SELECT task_id FROM memory_consolidation_jobs WHERE namespace=? AND status='running' ORDER BY created_at DESC LIMIT 1", (namespace,)).fetchone()
        if task:
            self.connection.execute("INSERT OR IGNORE INTO memory_task_sources VALUES(?,?,?)", (namespace, record_id, task[0]))
            self.connection.execute("UPDATE memory_consolidation_jobs SET source_ids_json=json_insert(source_ids_json,'$[#]',?),new_ids_json=json_insert(new_ids_json,'$[#]',?) WHERE task_id=?",
                                    (record_id, record_id, task[0]))

    def capture_job(self) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(
                """SELECT * FROM memory_capture_state WHERE cursor_id<requested_id AND due_at<=?
                   ORDER BY due_at,namespace LIMIT 1""", (int(time.time()),),
            ).fetchone()

    def capture_messages(self, namespace: int, after_id: int, through_id: int, limit: int) -> list[dict]:
        table, key, target = ("messages", "group_id", namespace) if namespace > 0 else ("private_messages", "user_id", -namespace)
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM {table} WHERE {key}=? AND id>? AND id<=? ORDER BY id LIMIT ?",
                (target, after_id, through_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def capture_retry(self, namespace: int, delay: int) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE memory_capture_state SET attempts=attempts+1,due_at=? WHERE namespace=?",
                (int(time.time()) + delay, namespace),
            )

    def apply_memory_capture(self, namespace: int, cursor_id: int | None, through_id: int, actions: list[dict]) -> None:
        """证据、版本、分层记录与索引任务在同一个SQLite事务提交。"""
        return self.memories.apply(namespace, cursor_id, through_id, actions)

    def forget_memory(self, namespace: int, subject: int, memory_id: int) -> bool:
        return self.memories.revoke(namespace, subject, memory_id)

    def group_context_rows(self, group_id: int, limit: int, before_id: int | None = None) -> list[dict]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM messages WHERE group_id=? AND (? IS NULL OR id<? OR role='assistant') ORDER BY id DESC LIMIT ?",
                (group_id, before_id, before_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def group_context_around(self, group_id: int, storage_id: int, before_id: int) -> list[dict]:
        """个人旧发言补上前后相邻消息，避免只传一方的半段对话。"""
        with self._lock:
            left = self.connection.execute(
                "SELECT * FROM messages WHERE group_id=? AND id<=? ORDER BY id DESC LIMIT 2", (group_id, storage_id),
            ).fetchall()
            right = self.connection.execute(
                "SELECT * FROM messages WHERE group_id=? AND id>? AND id<? ORDER BY id LIMIT 2", (group_id, storage_id, before_id),
            ).fetchall()
        return [dict(row) for row in reversed(left)] + [dict(row) for row in right]

    def group_context_message(self, group_id: int, message_id: str) -> dict | None:
        if not message_id:
            return None
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM messages WHERE group_id=? AND message_id=? ORDER BY id DESC LIMIT 1", (group_id, message_id),
            ).fetchone()
        return dict(row) if row and row["id"] not in self.memories.blocked_sources(group_id) else None

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
        self, group_id: int, user_id: int, nickname: str, content: str, *, message_id: str = "",
        role: str = "user", reply_to_message_id: str = "", mentions: list[int] | None = None,
    ) -> int:
        """保存群文本；assistant的reply_to字段记录逻辑回应对象，不代表实际显示QQ引用。"""
        if not content:
            return 0
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """INSERT INTO messages(group_id,user_id,nickname,content,created_at,message_id,role,reply_to_message_id,mentions_json)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (group_id, user_id, nickname, content[:4000], int(time.time()), message_id, role, reply_to_message_id,
                 json.dumps(mentions or [])),
            )
            record_id = int(cursor.lastrowid)
            if role in {"user", "assistant"} and not content.lstrip().startswith(("/", "[戳一戳", "[表情包")):
                self._queue_memory_capture(group_id, record_id)
                if role == "assistant":
                    self._assign_memory_reply(group_id, record_id, reply_to_message_id)
        self._notify_message(group_id, record_id)
        return record_id

    def _notify_message(self, namespace: int, record_id: int) -> None:
        if self.message_listener is not None:
            try:
                self.message_listener(namespace, record_id)
            except Exception as exc:
                logging.getLogger(__name__).warning("Working memory receive failed: error=%s", type(exc).__name__)

    def upsert_group_member(self, group_id: int, user_id: int, display_name: str, role: str = "member") -> None:
        """以群号和 QQ 号维护稳定成员身份，并保留最近使用过的昵称。"""
        display_name = display_name.strip()[:100] or str(user_id)
        now = int(time.time())
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT display_name,aliases_json FROM group_members WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            ).fetchone()
            aliases: list[str] = []
            if row is not None:
                try:
                    aliases = [str(value) for value in json.loads(row["aliases_json"]) if str(value).strip()]
                except (TypeError, ValueError):
                    aliases = []
                previous = row["display_name"].strip()
                if previous and previous != display_name and previous not in aliases:
                    aliases.append(previous)
            aliases = [alias for alias in aliases if alias != display_name][-10:]
            self.connection.execute(
                """INSERT INTO group_members
                   (group_id,user_id,display_name,aliases_json,role,first_seen_at,last_seen_at)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(group_id,user_id) DO UPDATE SET
                   display_name=excluded.display_name,aliases_json=excluded.aliases_json,
                   role=excluded.role,last_seen_at=excluded.last_seen_at""",
                (group_id, user_id, display_name, json.dumps(aliases, ensure_ascii=False), role, now, now),
            )

    def group_members(self, group_id: int, user_ids: set[int]) -> list[dict]:
        """查询本轮相关成员；昵称只展示，身份始终由 QQ 号确定。"""
        if not user_ids:
            return []
        placeholders = ",".join("?" for _ in user_ids)
        with self._lock:
            rows = list(self.connection.execute(
                f"""SELECT user_id,display_name,aliases_json,role,last_seen_at FROM group_members
                    WHERE group_id=? AND user_id IN ({placeholders}) ORDER BY user_id""",
                (group_id, *sorted(user_ids)),
            ))
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["aliases"] = json.loads(item.pop("aliases_json"))
            except (TypeError, ValueError):
                item["aliases"] = []
            result.append(item)
        return result

    def message_by_onebot_id(self, group_id: int, message_id: str) -> sqlite3.Row | None:
        """解析 QQ 引用目标；只返回同群消息，防止跨群关联。"""
        if not message_id:
            return None
        with self._lock:
            return self.connection.execute(
                """SELECT id,user_id,nickname,content,message_id,role,reply_to_message_id,created_at
                   FROM messages WHERE group_id=? AND message_id=? ORDER BY id DESC LIMIT 1""",
                (group_id, message_id),
            ).fetchone()

    def recent_user_messages(
        self, group_id: int, user_id: int, limit: int, *, before_id: int | None = None
    ) -> list[sqlite3.Row]:
        """补充当前成员自己的近期发言，避免被活跃群消息挤出上下文。"""
        if limit <= 0:
            return []
        boundary = " AND id<?" if before_id is not None else ""
        params = (group_id, user_id, before_id, limit) if before_id is not None else (group_id, user_id, limit)
        with self._lock:
            rows = list(self.connection.execute(
                """SELECT id,user_id,nickname,content,message_id,role,reply_to_message_id,created_at
                   FROM messages WHERE group_id=? AND user_id=?""" + boundary + " ORDER BY id DESC LIMIT ?",
                params,
            ))
        rows.reverse()
        return rows

    def memory_candidates(self, group_id: int, user_ids: set[int], limit: int) -> list[sqlite3.Row]:
        """给后台提取器提供本会话相关人物的既有事实；不是召回的最近N条截断。"""
        targets = sorted(user_ids | {0})
        placeholders = ",".join("?" for _ in targets)
        with self._lock:
            return list(self.connection.execute(
                f"""SELECT * FROM memory_records WHERE group_id=? AND user_id IN ({placeholders})
                    AND status='active' AND (expires_at=0 OR expires_at>strftime('%s','now'))
                    ORDER BY updated_at DESC LIMIT ?""",
                (group_id, *targets, limit),
            ))

    def memories_needing_embedding(self, embedding_model: str, limit: int) -> list[sqlite3.Row]:
        """为后台迁移读取尚未使用当前模型的记忆，不改变聊天请求的延迟。"""
        with self._lock:
            return list(self.connection.execute(
                """SELECT id,content FROM memory_records WHERE embedding_model<>? AND status IN ('active','candidate')
                   ORDER BY updated_at DESC,id DESC LIMIT ?""",
                (embedding_model, limit),
            ))

    def update_memory_embedding(
        self, memory_id: int, vector: str, embedding_model: str, embedding_dimensions: int, expected_content: str | None = None
    ) -> bool:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """UPDATE memory_records SET vector=?,embedding_model=?,embedding_dimensions=? WHERE id=?
                   AND status IN ('active','candidate') AND (? IS NULL OR content=?)""",
                (vector, embedding_model, embedding_dimensions, memory_id, expected_content, expected_content),
            )
            return cursor.rowcount > 0

    def touch_memories(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock, self.connection:
            self.connection.execute(
                f"UPDATE memory_records SET last_accessed_at=? WHERE id IN ({placeholders})",
                (int(time.time()), *memory_ids),
            )

    def recent_messages(self, group_id: int, limit: int, *, before_id: int | None = None) -> list[sqlite3.Row]:
        """取当前群最新的 limit 条消息，再按从旧到新的顺序返回给模型。"""
        with self._lock:
            boundary = " AND (id < ? OR role='assistant')" if before_id is not None else ""
            params = (group_id, before_id, limit) if before_id is not None else (group_id, limit)
            rows = list(self.connection.execute(
                "SELECT id,user_id,nickname,content,message_id,role,reply_to_message_id,created_at FROM messages"
                f" WHERE group_id=?{boundary} ORDER BY id DESC LIMIT ?", params,
            ))
        # 倒序查询便于限制最新条数，反转后符合聊天上下文的时间顺序。
        rows.reverse()
        return rows

    def recent_bot_messages(self, group_id: int, limit: int) -> list[sqlite3.Row]:
        """独立读取本群最近的机器人发言，避免被大量群友消息挤出窗口。"""
        if limit <= 0:
            return []
        with self._lock:
            rows = list(self.connection.execute(
                """SELECT id,user_id,nickname,content,message_id,role,reply_to_message_id,created_at
                   FROM messages WHERE group_id=? AND role='assistant' ORDER BY id DESC LIMIT ?""",
                (group_id, limit),
            ))
        return list(reversed(rows))

    def is_bot_message(self, group_id: int, message_id: str) -> bool:
        if not message_id:
            return False
        with self._lock:
            return self.connection.execute(
                "SELECT 1 FROM messages WHERE group_id=? AND message_id=? AND role='assistant' LIMIT 1",
                (group_id, message_id),
            ).fetchone() is not None

    def log_private_message(self, user_id: int, role: str, content: str) -> int:
        """保存私聊用户文本或已提交发送的机器人回复，每条最多 4000 字。"""
        if not content:
            return 0
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO private_messages(user_id,role,content,created_at) VALUES(?,?,?,?)",
                (user_id, role, content[:4000], int(time.time())),
            )
            if not content.lstrip().startswith(("/", "[戳一戳", "[表情包")):
                self._queue_memory_capture(-user_id, int(cursor.lastrowid))
                if role == "assistant":
                    self._assign_memory_reply(-user_id, int(cursor.lastrowid))
            record_id = int(cursor.lastrowid)
        self._notify_message(-user_id, record_id)
        return record_id

    def recent_private_messages(self, user_id: int, limit: int) -> list[sqlite3.Row]:
        """只读取该用户与机器人的私聊，按时间顺序返回角色和文本。"""
        with self._lock:
            rows = list(self.connection.execute(
                """SELECT id,role,content FROM private_messages
                   WHERE user_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ))
        rows.reverse()
        blocked = self.memories.blocked_sources(-user_id)
        return [row for row in rows if row["id"] not in blocked]

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
            self.memories.clear_user(-user_id, user_id)
            messages = self.connection.execute("DELETE FROM private_messages WHERE user_id=?", (user_id,))
            reminders = self.connection.execute("DELETE FROM reminders WHERE user_id=?", (user_id,))
            self.connection.execute("DELETE FROM memory_records WHERE group_id=?", (-user_id,))
            self.connection.execute("DELETE FROM memory_capture_state WHERE namespace=?", (-user_id,))
            return messages.rowcount, reminders.rowcount

    def memory_source_rows(self, namespace: int, ids: set[int]) -> list[dict]:
        if not ids:
            return []
        table, key, target = ("messages", "group_id", namespace) if namespace > 0 else ("private_messages", "user_id", -namespace)
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM {table} WHERE {key}=? AND id IN ({','.join('?' for _ in ids)}) ORDER BY id",
                (target, *sorted(ids)),
            ).fetchall()
        return [dict(row) for row in rows]

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
        """取目标用户在当前群的普通文字；排除动作通知、机器人回复、空文本及 / 指令。"""
        with self._lock:
            rows = list(
                self.connection.execute(
                    """SELECT content FROM messages
                       WHERE group_id=? AND user_id=? AND role='user'
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

    def delete_group_user_data(self, group_id: int, user_id: int) -> tuple[int, int]:
        """原子删除本人在当前群的消息、成员档案及作为本人来源的长期记忆。"""
        with self._lock, self.connection:
            memory_count = self.connection.execute("SELECT count(*) FROM memory_records WHERE group_id=?", (group_id,)).fetchone()[0]
            self.memories.clear_user(group_id, user_id)
            messages = self.connection.execute(
                "DELETE FROM messages WHERE group_id=? AND user_id=?", (group_id, user_id)
            )
            memories = self.connection.execute(
                "DELETE FROM memory_records WHERE group_id=? AND (user_id=? OR source_user_id=?)",
                (group_id, user_id, user_id),
            )
            self.connection.execute(
                "DELETE FROM group_members WHERE group_id=? AND user_id=?", (group_id, user_id)
            )
            remaining = self.connection.execute("SELECT count(*) FROM memory_records WHERE group_id=?", (group_id,)).fetchone()[0]
            return messages.rowcount, memory_count - remaining

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

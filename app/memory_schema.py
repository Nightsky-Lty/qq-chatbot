"""分层记忆权威表的初始化；仅支持当前存储结构。"""

import sqlite3


# memory_type是层内子类型，不是第二套记忆系统。
COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "group_id": "INTEGER NOT NULL",
    "user_id": "INTEGER NOT NULL",
    "source_user_id": "INTEGER NOT NULL",
    "memory_type": "TEXT NOT NULL",
    "content": "TEXT NOT NULL",
    "vector": "TEXT NOT NULL DEFAULT '{}'",
    "hash_vector": "TEXT NOT NULL DEFAULT '{}'",
    "embedding_model": "TEXT NOT NULL DEFAULT 'hash-blake2b-512-v1'",
    "embedding_dimensions": "INTEGER NOT NULL DEFAULT 512",
    "importance": "REAL NOT NULL DEFAULT 0.5",
    "confidence": "REAL NOT NULL DEFAULT 0.8",
    "source_message_id": "TEXT NOT NULL DEFAULT ''",
    "created_at": "INTEGER NOT NULL",
    "updated_at": "INTEGER NOT NULL",
    "last_accessed_at": "INTEGER NOT NULL DEFAULT 0",
    "status": "TEXT NOT NULL DEFAULT 'active'",
    "fact_key": "TEXT NOT NULL DEFAULT ''",
    "sources_json": "TEXT NOT NULL DEFAULT '[]'",
    "expires_at": "INTEGER NOT NULL DEFAULT 0",
    "layer": "TEXT NOT NULL DEFAULT 'abstract' CHECK(layer IN ('event','abstract','perceptual'))",
    "version": "INTEGER NOT NULL DEFAULT 1",
    "occurred_at": "INTEGER NOT NULL DEFAULT 0",
    "evidence_kind": "TEXT NOT NULL DEFAULT 'reported'",
    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
}


def initialize_memory_schema(conn: sqlite3.Connection) -> None:
    """幂等创建当前记忆表，不检测、导入或拦截旧数据库。"""
    fields = ",".join(f"{name} {definition}" for name, definition in COLUMNS.items())
    with conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS memory_records({fields}, UNIQUE(group_id,user_id,memory_type,content))")

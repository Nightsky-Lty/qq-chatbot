"""分层记忆的唯一权威表，以及只在升级时运行的旧库导入器。"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

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


def _create_records(conn):
    fields = ",".join(f"{name} {definition}" for name, definition in COLUMNS.items())
    conn.execute(f"CREATE TABLE memory_records({fields}, UNIQUE(group_id,user_id,memory_type,content))")


def _backup(conn, path):
    directory = path.parent / "backups"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"before-memory-records-{stamp}-{uuid.uuid4().hex[:8]}.db"
    # 独立文件，SQLite backup API也包含WAL中的已提交数据；失败禁止继续迁移。
    target.touch(mode=0o600, exist_ok=False)
    with sqlite3.connect(target) as destination:
        conn.backup(destination)
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("记忆迁移备份校验失败")
    logger.info("Memory storage migration backup ready: path=%s", target)
    return target


def initialize_memory_schema(conn: sqlite3.Connection, path: Path):
    """备份→事务复制/逐字段核对→移除旧表；异常回滚，不双写、不吞冲突。"""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "memories" not in tables:
        if "memory_records" not in tables:
            with conn:
                _create_records(conn)
        return None
    if "memory_records" in tables:
        raise RuntimeError("新旧记忆表同时存在，拒绝自动合并：请先人工核对编号冲突")
    backup = _backup(conn, path)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    if columns - COLUMNS.keys():
        raise RuntimeError("旧记忆表有未知字段，已备份，停止迁移以免丢失数据")
    required = {"id", "group_id", "user_id", "source_user_id", "memory_type", "content", "created_at", "updated_at"}
    if not required <= columns:
        raise RuntimeError("旧记忆表缺少必要字段，已备份，停止迁移")
    # 不在此处调用LLM、改变事实、激活失效记忆或推断不存在的来源。
    expressions = {}
    for name, definition in COLUMNS.items():
        if name in columns:
            expressions[name] = name
        elif " DEFAULT " in definition:
            expressions[name] = definition.split(" DEFAULT ", 1)[1].split(" CHECK", 1)[0]
        else:
            raise RuntimeError(f"旧记忆缺少字段：{name}")
    if "hash_vector" not in columns:
        expressions["hash_vector"] = f"CASE WHEN {expressions['embedding_model']}='hash-blake2b-512-v1' THEN {expressions['vector']} ELSE '{{}}' END"
    else:
        expressions["hash_vector"] = f"CASE WHEN hash_vector IN ('','{{}}') AND {expressions['embedding_model']}='hash-blake2b-512-v1' THEN {expressions['vector']} ELSE hash_vector END"
    if "layer" not in columns:
        expressions["layer"] = "CASE WHEN memory_type IN ('event','episode','commitment') THEN 'event' WHEN memory_type='image' THEN 'perceptual' ELSE 'abstract' END"
    else:
        expressions["layer"] = "CASE WHEN memory_type IN ('event','episode','commitment') AND layer='abstract' THEN 'event' WHEN memory_type='image' AND layer='abstract' THEN 'perceptual' ELSE layer END"
    if "occurred_at" not in columns:
        expressions["occurred_at"] = "created_at"
    names = ",".join(COLUMNS)
    projection = ",".join(expressions.values())
    try:
        conn.execute("BEGIN IMMEDIATE")
        _create_records(conn)
        conn.execute(f"INSERT INTO memory_records({names}) SELECT {projection} FROM memories")
        # 两个方向比较所有字段，不仅仅核对总条数。
        for left, right in ((f"SELECT {projection} FROM memories", f"SELECT {names} FROM memory_records"),
                            (f"SELECT {names} FROM memory_records", f"SELECT {projection} FROM memories")):
            if conn.execute(f"{left} EXCEPT {right} LIMIT 1").fetchone():
                raise RuntimeError("新旧记忆逐字段校验不一致")
        count = 0
        for row in conn.execute("SELECT id,sources_json,metadata_json FROM memory_records"):
            sources, metadata = json.loads(row[1]), json.loads(row[2])
            if (not isinstance(sources, list) or not isinstance(metadata, dict)
                    or any(not isinstance(s, dict) or type(s.get("id")) is not int for s in sources)):
                raise ValueError("旧记忆来源或元数据格式不正确")
            count += 1
            metadata["storage_migration"] = {"from": "memories", "version": 1,
                                             "has_source_excerpts": any(isinstance(s.get("content"), str) and s["content"].strip() for s in sources)}
            conn.execute("UPDATE memory_records SET metadata_json=? WHERE id=?", (json.dumps(metadata, ensure_ascii=False), row[0]))
        # 保留已删除编号的高水位，避免新记录复用旧Qdrant点位或任务引用。
        sequence = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='memories'").fetchone()
        if sequence:
            conn.execute("UPDATE sqlite_sequence SET seq=max(seq,?) WHERE name='memory_records'", (sequence[0],))
            if not count:
                conn.execute("INSERT INTO sqlite_sequence(name,seq) SELECT 'memory_records',? WHERE NOT EXISTS(SELECT 1 FROM sqlite_sequence WHERE name='memory_records')", (sequence[0],))
        conn.execute("DROP VIEW IF EXISTS memory_events")
        conn.execute("DROP VIEW IF EXISTS memory_concepts")
        conn.execute("DROP VIEW IF EXISTS memory_percepts")
        # DROP TABLE不会逐条触发DELETE，不会误删关系或把所有向量排队删除。
        conn.execute("DROP TABLE memories")
        if "memory_schema_meta" in tables:
            conn.execute("DELETE FROM memory_schema_meta WHERE key='fts_v1'")
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise RuntimeError("迁移后的外键校验失败")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    logger.info("Memory storage migration complete: records=%s backup=%s", count, backup)
    return backup

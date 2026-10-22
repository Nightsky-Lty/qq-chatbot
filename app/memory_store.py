"""分层记忆权威存储：证据、FTS、版本、遗忘和可恢复任务。"""

import hashlib
import json
import re
import time
from typing import TYPE_CHECKING

from app.embeddings import tokens

if TYPE_CHECKING:
    from app.database import Database


def fingerprint(value: str) -> str:
    return hashlib.sha256("".join(value.casefold().split()).encode()).hexdigest()


def terms(value: str) -> str:
    return " ".join(tokens(value))


def match_query(value: str) -> str:
    # 不把用户文本当FTS语法；中文双字片段、英文单词均可命中。
    words = list(dict.fromkeys(t for t in tokens(value) if len(t) > 1))[:40]
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in words)


class MemoryStore:
    def __init__(self, db: "Database"):
        self.db = db
        conn = db.connection
        conn.create_function("memory_terms", 1, terms, deterministic=True)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_revisions(
          memory_id INTEGER NOT NULL, version INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
          created_at INTEGER NOT NULL, PRIMARY KEY(memory_id,version));
        CREATE TABLE IF NOT EXISTS memory_links(
          namespace INTEGER NOT NULL, source_id INTEGER NOT NULL, target_id INTEGER NOT NULL,
          relation TEXT NOT NULL, PRIMARY KEY(namespace,source_id,target_id,relation));
        CREATE TABLE IF NOT EXISTS memory_blocks(
          namespace INTEGER NOT NULL, subject INTEGER NOT NULL, key_hash TEXT NOT NULL,
          content_hash TEXT NOT NULL, source_ids_json TEXT NOT NULL, created_at INTEGER NOT NULL,
          PRIMARY KEY(namespace,subject,key_hash,content_hash));
        CREATE TABLE IF NOT EXISTS memory_index_jobs(
          memory_id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          due_at INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS memory_consolidation_jobs(
          task_id TEXT PRIMARY KEY, namespace INTEGER NOT NULL, source_ids_json TEXT NOT NULL,
          new_ids_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          due_at INTEGER NOT NULL, created_at INTEGER NOT NULL, result_json TEXT NOT NULL DEFAULT '{}');
        CREATE INDEX IF NOT EXISTS idx_memory_jobs_due ON memory_consolidation_jobs(status,due_at);
        CREATE TABLE IF NOT EXISTS memory_task_sources(
          namespace INTEGER NOT NULL,source_id INTEGER NOT NULL,task_id TEXT NOT NULL,
          PRIMARY KEY(namespace,source_id));
        CREATE TABLE IF NOT EXISTS memory_media_links(
          namespace INTEGER NOT NULL,event_id INTEGER NOT NULL,ref_id INTEGER NOT NULL,
          PRIMARY KEY(namespace,event_id,ref_id));
        CREATE INDEX IF NOT EXISTS idx_memory_layer_scope ON memory_records(group_id,layer,status);
        CREATE INDEX IF NOT EXISTS idx_memory_records_owner ON memory_records(group_id,user_id,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_records_source ON memory_records(group_id,source_user_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(terms);
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_source_fts USING fts5(
          namespace UNINDEXED, source_id UNINDEXED, terms);
        CREATE TABLE IF NOT EXISTS memory_schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE VIEW IF NOT EXISTS memory_events AS SELECT * FROM memory_records WHERE layer='event';
        CREATE VIEW IF NOT EXISTS memory_concepts AS SELECT * FROM memory_records WHERE layer='abstract';
        CREATE VIEW IF NOT EXISTS memory_percepts AS SELECT * FROM memory_records WHERE layer='perceptual';
        CREATE TRIGGER IF NOT EXISTS memory_insert_index AFTER INSERT ON memory_records BEGIN
          INSERT INTO memory_fts(rowid,terms) VALUES(new.id,memory_terms(new.content||' '||new.fact_key));
          INSERT OR REPLACE INTO memory_index_jobs(memory_id,revision) VALUES(new.id,new.version);
        END;
        CREATE TRIGGER IF NOT EXISTS memory_update_index AFTER UPDATE OF content,fact_key,status,vector,embedding_model,metadata_json,version ON memory_records BEGIN
          DELETE FROM memory_fts WHERE rowid=old.id;
          INSERT INTO memory_fts(rowid,terms) VALUES(new.id,memory_terms(new.content||' '||new.fact_key));
          INSERT OR REPLACE INTO memory_index_jobs(memory_id,revision) VALUES(new.id,new.version);
        END;
        CREATE TRIGGER IF NOT EXISTS memory_delete_index AFTER DELETE ON memory_records BEGIN
          DELETE FROM memory_fts WHERE rowid=old.id;
          INSERT OR REPLACE INTO memory_index_jobs(memory_id,revision) VALUES(old.id,old.version+1);
          DELETE FROM memory_links WHERE source_id=old.id OR target_id=old.id;
        END;
        CREATE TRIGGER IF NOT EXISTS memory_group_source_insert AFTER INSERT ON messages BEGIN
          INSERT INTO memory_source_fts(namespace,source_id,terms) VALUES(new.group_id,new.id,memory_terms(new.content));
        END;
        CREATE TRIGGER IF NOT EXISTS memory_group_source_delete AFTER DELETE ON messages BEGIN
          DELETE FROM memory_source_fts WHERE namespace=old.group_id AND source_id=old.id;
        END;
        CREATE TRIGGER IF NOT EXISTS memory_private_source_insert AFTER INSERT ON private_messages BEGIN
          INSERT INTO memory_source_fts(namespace,source_id,terms) VALUES(-new.user_id,new.id,memory_terms(new.content));
        END;
        CREATE TRIGGER IF NOT EXISTS memory_private_source_delete AFTER DELETE ON private_messages BEGIN
          DELETE FROM memory_source_fts WHERE namespace=-old.user_id AND source_id=old.id;
        END;
        """)
        with db._lock, conn:
            if conn.execute("SELECT 1 FROM memory_schema_meta WHERE key='fts_v1'").fetchone() is None:
                conn.execute("DELETE FROM memory_fts")
                conn.execute("INSERT INTO memory_fts(rowid,terms) SELECT id,memory_terms(content||' '||fact_key) FROM memory_records")
                conn.execute("DELETE FROM memory_source_fts")
                conn.execute("INSERT INTO memory_source_fts SELECT group_id,id,memory_terms(content) FROM messages")
                conn.execute("INSERT INTO memory_source_fts SELECT -user_id,id,memory_terms(content) FROM private_messages")
                conn.execute("INSERT OR IGNORE INTO memory_index_jobs(memory_id,revision) SELECT id,version FROM memory_records")
                conn.execute("INSERT INTO memory_schema_meta VALUES('fts_v1','1')")
            if conn.execute("SELECT 1 FROM memory_schema_meta WHERE key='forgotten_redaction_v1'").fetchone() is None:
                for row in conn.execute("SELECT * FROM memory_records WHERE status='forgotten'").fetchall():
                    sources = [s.get("id") for s in json.loads(row["sources_json"]) if type(s.get("id")) is int]
                    conn.execute("INSERT OR IGNORE INTO memory_blocks VALUES(?,?,?,?,?,?)", (row["group_id"], row["user_id"],
                        fingerprint(row["fact_key"] + "|"), fingerprint(row["content"]), json.dumps(sources), int(time.time())))
                    conn.execute("UPDATE memory_records SET content=?,fact_key='',sources_json='[]',vector='{}',hash_vector='{}',metadata_json='{}',version=version+1 WHERE id=?",
                                 (f"[已遗忘:{row['id']}]", row["id"]))
                conn.execute("INSERT INTO memory_schema_meta VALUES('forgotten_redaction_v1','1')")

    @staticmethod
    def layer(kind: str) -> str:
        return "event" if kind in {"event", "episode", "commitment"} else "perceptual" if kind == "image" else "abstract"

    def row(self, namespace: int, memory_id: int, *, active: bool = True):
        with self.db._lock:
            row = self.db.connection.execute("SELECT * FROM memory_records WHERE group_id=? AND id=?", (namespace, memory_id)).fetchone()
            if row is None or (active and (row["status"] != "active" or (row["expires_at"] and row["expires_at"] <= time.time()))):
                return None
            return dict(row)

    def blocked(self, namespace: int, subject: int, key: str, content: str, source_ids=(), conditions="") -> bool:
        with self.db._lock:
            rows = self.db.connection.execute("SELECT * FROM memory_blocks WHERE namespace=?", (namespace,)).fetchall()
        ids = set(source_ids)
        return any((r["subject"] == subject and (r["key_hash"] == fingerprint(key + "|" + conditions) or r["content_hash"] == fingerprint(content)))
                   or bool(ids & set(json.loads(r["source_ids_json"]))) for r in rows)

    def blocked_sources(self, namespace: int) -> set[int]:
        with self.db._lock:
            rows = self.db.connection.execute("SELECT source_ids_json FROM memory_blocks WHERE namespace=?", (namespace,)).fetchall()
        return {i for row in rows for i in json.loads(row[0])}

    def remember_revision(self, row: dict):
        self.db.connection.execute("INSERT OR IGNORE INTO memory_revisions VALUES(?,?,?,?)",
                                   (row["id"], row["version"], json.dumps(row, ensure_ascii=False), int(time.time())))

    def revoke(self, namespace: int, subject: int, memory_id: int, *, delete: bool = False, expected_version: int | None = None) -> bool:
        with self.db._lock, self.db.connection:
            row = self.row(namespace, memory_id, active=False)
            if row is None or row["user_id"] != subject or row["status"] not in {"active", "candidate", "superseded", "archived"}:
                return False
            if expected_version is not None and expected_version != row["version"]:
                raise ValueError("记忆版本已变化，请重新搜索后操作")
            # 同一事实旧版本和由其派生的概念一并停用，不能从来源/旧版本绕过遗忘。
            related = {memory_id}
            changed = True
            while changed:
                before = len(related)
                for link in self.db.connection.execute("SELECT * FROM memory_links WHERE namespace=?", (namespace,)):
                    if link["relation"] == "derived_from" and link["target_id"] in related:
                        related.add(link["source_id"])
                    if link["relation"] == "revises" and (link["source_id"] in related or link["target_id"] in related):
                        related.update((link["source_id"], link["target_id"]))
                changed = len(related) != before
            for target in related:
                item = self.row(namespace, target, active=False)
                if not item:
                    continue
                sources = [s.get("id") for s in json.loads(item["sources_json"]) if type(s.get("id")) is int]
                # 同时阻断对这些原话的直接机器人复述，避免从聊天历史绕回来。
                if namespace > 0 and sources:
                    originals = self.db.memory_source_rows(namespace, set(sources))
                    message_ids = [s.get("message_id") for s in originals if s.get("message_id")]
                    if message_ids:
                        sources.extend(r[0] for r in self.db.connection.execute(
                            "SELECT id FROM messages WHERE group_id=? AND role='assistant' AND reply_to_message_id IN ("
                            + ",".join("?" for _ in message_ids) + ")", (namespace, *message_ids)))
                elif namespace < 0:
                    for source_id in list(sources):
                        next_user = self.db.connection.execute("SELECT min(id) FROM private_messages WHERE user_id=? AND role='user' AND id>?", (-namespace, source_id)).fetchone()[0]
                        sources.extend(r[0] for r in self.db.connection.execute(
                            "SELECT id FROM private_messages WHERE user_id=? AND role='assistant' AND id>? AND (? IS NULL OR id<?) LIMIT 10",
                            (-namespace, source_id, next_user, next_user)))
                sources = sorted(set(sources))
                self.db.connection.execute("INSERT OR IGNORE INTO memory_blocks VALUES(?,?,?,?,?,?)",
                    (namespace, item["user_id"], fingerprint(item["fact_key"] + "|" + json.loads(item["metadata_json"]).get("conditions", ""))
                     if item["layer"] == "abstract" else fingerprint(""), fingerprint(item["content"]), json.dumps(sources), int(time.time())))
                self.db.connection.execute("DELETE FROM memory_revisions WHERE memory_id=?", (target,))
                if delete:
                    self.db.connection.execute("DELETE FROM memory_records WHERE id=?", (target,))
                else:
                    self.db.connection.execute("""UPDATE memory_records SET status='forgotten',content=?,fact_key='',sources_json='[]',
                      metadata_json='{}',vector='{}',hash_vector='{}',version=version+1,updated_at=? WHERE id=?""",
                      (f"[已遗忘:{target}]", int(time.time()), target))
                self.db.connection.execute("DELETE FROM memory_links WHERE namespace=? AND (source_id=? OR target_id=?)", (namespace, target, target))
            return True

    def apply(self, namespace: int, cursor_id: int | None, through_id: int, actions: list[dict]):
        conn, now = self.db.connection, int(time.time())
        stats = {"stored": 0, "candidates": 0, "duplicates": 0, "updated": 0}
        table, key, owner = ("messages", "group_id", namespace) if namespace > 0 else ("private_messages", "user_id", -namespace)
        with self.db._lock, conn:
            state = conn.execute("SELECT cursor_id FROM memory_capture_state WHERE namespace=?", (namespace,)).fetchone()
            if cursor_id is not None and (state is None or state[0] != cursor_id):
                raise ValueError("记忆任务状态已变化")
            for action in actions:
                for source_id in action["source_ids"]:
                    if conn.execute(f"SELECT 1 FROM {table} WHERE {key}=? AND id=?", (owner, source_id)).fetchone() is None:
                        raise ValueError("记忆来源已删除")
                subject = action["subject_user_id"]
                if self.blocked(namespace, subject, action["fact_key"], action["content"], action["source_ids"], action.get("metadata", {}).get("conditions", "")):
                    raise ValueError("来源或事实已被遗忘，不允许后台重建")
                for old_id in action["replaces_ids"]:
                    old = self.row(namespace, old_id)
                    if not old or old["user_id"] != subject or old["version"] != action.get("expected_versions", {}).get(old_id, old["version"]):
                        raise ValueError("旧记忆版本已变化")
                    self.remember_revision(old)
                    conn.execute("UPDATE memory_records SET status='superseded',version=version+1,updated_at=? WHERE id=?", (now, old_id))
                if action["action"] == "forget":
                    # 自动整理不拥有删除/遗忘权限；显式操作走独立绑定工具。
                    raise ValueError("遗忘必须使用本人授权的专用工具")
                metadata = action.get("metadata", {})
                duplicate = conn.execute("SELECT * FROM memory_records WHERE group_id=? AND user_id=? AND memory_type=? AND content=?",
                                         (namespace, subject, action["memory_type"], action["content"])).fetchone()
                if duplicate and self.layer(action["memory_type"]) == "event" and duplicate["occurred_at"] != action.get("occurred_at", 0):
                    # 正文唯一键用于幂等；相同描述在不同时间发生不能被当成同一事件。
                    original = action["content"]
                    suffix = f" [事件时间:{action.get('occurred_at', 0)}]"
                    action["content"] = original[:max(2, getattr(self.db, "memory_max_chars", 200) - len(suffix))] + suffix
                    metadata["original_summary"] = original
                    duplicate = conn.execute("SELECT * FROM memory_records WHERE group_id=? AND user_id=? AND memory_type=? AND content=?",
                                             (namespace, subject, action["memory_type"], action["content"])).fetchone()
                    if self.blocked(namespace, subject, action["fact_key"], action["content"], action["source_ids"], metadata.get("conditions", "")):
                        raise ValueError("该事件版本已被遗忘")
                values = (namespace, subject, subject, action["memory_type"], action["content"], action["vector"], action["hash_vector"],
                          action["embedding_model"], action["embedding_dimensions"], action["importance"], action["confidence"],
                          str(action["source_ids"][0]), now, now, action["fact_key"], json.dumps(action["sources"], ensure_ascii=False),
                          action["expires_at"], self.layer(action["memory_type"]), action.get("status", "active"),
                          action.get("occurred_at", 0), action.get("evidence_kind", "reported"), json.dumps(metadata, ensure_ascii=False))
                if duplicate:
                    # 重复证据只合并来源，不用反复检索/提取提高重要性或复活旧记录。
                    if duplicate["status"] not in {"active", "candidate"}:
                        stats["duplicates"] += 1
                        continue
                    sources = {s["id"]: s for s in json.loads(duplicate["sources_json"])}
                    sources.update({s["id"]: s for s in action["sources"]})
                    conn.execute("UPDATE memory_records SET sources_json=? WHERE id=?", (json.dumps(list(sources.values())[-24:], ensure_ascii=False), duplicate["id"]))
                    if duplicate["status"] == "candidate" and action.get("status", "active") == "active":
                        self.remember_revision(dict(duplicate))
                        conn.execute("UPDATE memory_records SET status='active',version=version+1,metadata_json=?,evidence_kind=?,updated_at=? WHERE id=?",
                            (json.dumps(metadata, ensure_ascii=False), action.get("evidence_kind", "reported"), now, duplicate["id"]))
                        stats["stored"] += 1
                    else:
                        stats["duplicates"] += 1
                    memory_id = duplicate["id"]
                else:
                    cur = conn.execute("""INSERT INTO memory_records(group_id,user_id,source_user_id,memory_type,content,vector,hash_vector,
                      embedding_model,embedding_dimensions,importance,confidence,source_message_id,created_at,updated_at,
                      fact_key,sources_json,expires_at,layer,status,occurred_at,evidence_kind,metadata_json)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
                    memory_id = cur.lastrowid
                    stats["candidates" if action.get("status", "active") == "candidate" else "stored"] += 1
                stats["updated"] += len(action["replaces_ids"])
                for old_id in action["replaces_ids"]:
                    conn.execute("INSERT OR IGNORE INTO memory_links VALUES(?,?,?,'revises')", (namespace, memory_id, old_id))
                for evidence_id in metadata.get("support_ids", []):
                    if self.row(namespace, evidence_id):
                        conn.execute("INSERT OR IGNORE INTO memory_links VALUES(?,?,?,'derived_from')", (namespace, memory_id, evidence_id))
                for ref_id in metadata.get("media_refs", []):
                    conn.execute("INSERT OR IGNORE INTO memory_media_links VALUES(?,?,?)", (namespace, memory_id, ref_id))
            for subject in {a["subject_user_id"] for a in actions}:
                conn.execute("""UPDATE memory_records SET status='archived',version=version+1 WHERE id IN
                  (SELECT id FROM memory_records WHERE group_id=? AND user_id=? AND status='active' AND layer!='perceptual'
                  ORDER BY importance DESC,updated_at DESC,id DESC LIMIT -1 OFFSET ?)""",
                  (namespace, subject, getattr(self.db, "memory_max_per_user", 200)))
            if cursor_id is not None:
                conn.execute("""UPDATE memory_capture_state SET cursor_id=?,attempts=0,
                  due_at=CASE WHEN requested_id>? THEN ? ELSE ? END WHERE namespace=?""",
                  (through_id, through_id, now, now + self.db.capture_delay, namespace))
        return stats

    def enqueue(self, task_id: str, namespace: int, source_ids: set[int], new_ids: set[int]) -> bool:
        if not source_ids or not new_ids:
            return False
        # 引用必须仍存在，工作区中的旧检索结果、模型草稿和推理不入队。
        valid = {r["id"] for r in self.db.memory_source_rows(namespace, source_ids)} - self.blocked_sources(namespace)
        fresh = valid & new_ids
        if not fresh:
            return False
        with self.db._lock, self.db.connection:
            cur = self.db.connection.execute("""INSERT INTO memory_consolidation_jobs(task_id,namespace,source_ids_json,new_ids_json,due_at,created_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET source_ids_json=excluded.source_ids_json,
              new_ids_json=excluded.new_ids_json,status=CASE WHEN status='running' THEN 'pending' ELSE status END""",
              (task_id, namespace, json.dumps(sorted(valid)), json.dumps(sorted(fresh)), int(time.time()), int(time.time())))
            self.db.connection.executemany("INSERT OR IGNORE INTO memory_task_sources VALUES(?,?,?)", [(namespace, i, task_id) for i in valid])
            return cur.rowcount > 0

    def clear_user(self, namespace: int, user_id: int):
        """在调用方事务内删除本人以及含本人来源的派生记忆和待整理任务。"""
        conn = self.db.connection
        for row in conn.execute("SELECT * FROM memory_records WHERE group_id=?", (namespace,)).fetchall():
            sources = json.loads(row["sources_json"])
            if row["user_id"] == user_id or any(s.get("user_id") == user_id for s in sources):
                conn.execute("DELETE FROM memory_revisions WHERE memory_id=?", (row["id"],))
                conn.execute("DELETE FROM memory_records WHERE id=?", (row["id"],))
        table, key, target = ("messages", "group_id", namespace) if namespace > 0 else ("private_messages", "user_id", -namespace)
        own_ids = {r[0] for r in conn.execute(f"SELECT id FROM {table} WHERE {key}=?" + (" AND user_id=?" if namespace > 0 else ""),
                   (target, user_id) if namespace > 0 else (target,))}
        for job in conn.execute("SELECT * FROM memory_consolidation_jobs WHERE namespace=?", (namespace,)).fetchall():
            remaining = set(json.loads(job["source_ids_json"])) - own_ids
            fresh = set(json.loads(job["new_ids_json"])) - own_ids
            if not fresh:
                conn.execute("DELETE FROM memory_consolidation_jobs WHERE task_id=?", (job["task_id"],))
            else:
                conn.execute("UPDATE memory_consolidation_jobs SET source_ids_json=?,new_ids_json=? WHERE task_id=?", (json.dumps(sorted(remaining)), json.dumps(sorted(fresh)), job["task_id"]))
        conn.executemany("DELETE FROM memory_task_sources WHERE namespace=? AND source_id=?", [(namespace, i) for i in own_ids])
        conn.execute("DELETE FROM memory_blocks WHERE namespace=? AND subject=?", (namespace, user_id))
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_cache_refs'").fetchone():
            conn.execute("UPDATE image_cache_refs SET expires_at=0,status='unavailable',image_json='{}' WHERE namespace=? AND user_id=?", (namespace, user_id))

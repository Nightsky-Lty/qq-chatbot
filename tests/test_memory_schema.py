import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.memory import MemoryService
from app.memory_schema import initialize_memory_schema


class MemorySchemaTests(unittest.TestCase):
    def test_initialization_only_creates_current_table_without_legacy_detection(self):
        with sqlite3.connect(":memory:") as conn:
            statements = []
            conn.set_trace_callback(statements.append)
            initialize_memory_schema(conn)
            initialize_memory_schema(conn)
            self.assertEqual(len(statements), 2)
            self.assertTrue(all(s.startswith("CREATE TABLE IF NOT EXISTS memory_records(") for s in statements))

    def test_fresh_database_and_views_are_ready_without_migration_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bot.db"
            db = Database(path)
            try:
                names = {r[0] for r in db.connection.execute("SELECT name FROM sqlite_master")}
                self.assertTrue({"memory_records", "memory_events", "memory_concepts", "memory_percepts"} <= names)
                self.assertNotIn("memories", names)
                self.assertFalse(hasattr(db, "memory_migration_backup"))
                self.assertFalse((path.parent / "backups").exists())
                self.assertIsNone(db.connection.execute("SELECT 1 FROM memory_schema_meta WHERE key='forgotten_redaction_v1'").fetchone())
                for name in ("save_memory", "delete_user_memory", "list_user_memories"):
                    self.assertFalse(hasattr(db, name))
            finally:
                db.close()

    def test_reopen_preserves_records_relations_and_high_watermark(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bot.db"
            db = Database(path)
            with db.connection as conn:
                for i, status in enumerate(("active", "candidate", "superseded", "forgotten", "archived"), 1):
                    conn.execute("""INSERT INTO memory_records(id,group_id,user_id,source_user_id,memory_type,
                        content,created_at,updated_at,version,status) VALUES(?,7,11,11,'preference',?,100,200,3,?)""",
                                 (i, f"内容{i}" if status != "forgotten" else f"[已遗忘:{i}]", status))
                conn.execute("INSERT INTO memory_revisions VALUES(1,2,'{}',100)")
                conn.execute("INSERT INTO memory_links VALUES(7,1,2,'derived_from')")
                conn.execute("INSERT INTO memory_media_links VALUES(7,1,12)")
                conn.execute("INSERT INTO memory_blocks VALUES(7,11,'key','content','[999]',100)")
                conn.execute("UPDATE sqlite_sequence SET seq=999 WHERE name='memory_records'")
            expected = [tuple(r) for r in db.connection.execute("SELECT * FROM memory_records ORDER BY id")]
            db.close()
            db = Database(path)
            try:
                self.assertEqual([tuple(r) for r in db.connection.execute("SELECT * FROM memory_records ORDER BY id")], expected)
                for table in ("memory_links", "memory_media_links", "memory_revisions", "memory_blocks"):
                    self.assertEqual(db.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 1)
                self.assertIsNone(db.memories.row(7, 4))
                self.assertIn(999, db.memories.blocked_sources(7))
                with db.connection:
                    cur = db.connection.execute("""INSERT INTO memory_records(group_id,user_id,source_user_id,
                        memory_type,content,created_at,updated_at) VALUES(7,11,11,'event','新事件',100,200)""")
                self.assertEqual(cur.lastrowid, 1000)
                self.assertTrue(db.connection.execute("SELECT 1 FROM memory_index_jobs WHERE memory_id=1000").fetchone())
                self.assertFalse((path.parent / "backups").exists())
            finally:
                db.close()


class MemorySchemaRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_data_remains_searchable_and_forgettable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bot.db"
            db = Database(path)
            # 保留已在线上数据中的元数据，但运行时不再读取迁移标记。
            with db.connection as conn:
                conn.execute("""INSERT INTO memory_records(id,group_id,user_id,source_user_id,memory_type,
                    content,created_at,updated_at,layer,metadata_json)
                    VALUES(17,7,22,22,'event','QQ22修复了火星网站连接错误',100,200,'event',?)""",
                             (json.dumps({"storage_migration": {"version": 1}}),))
            db.close()
            db = Database(path)
            cfg = Settings(_env_file=None, embedding_provider="hash", memory_qdrant_url="",
                           memory_source_search_enabled=False)
            memory = MemoryService(cfg, db)
            try:
                found = await memory.retrieve(7, "火星网站连接错误", user_ids={11}, layer="event")
                self.assertEqual(found[0]["记忆编号"], 17)
                self.assertEqual(found[0]["来源说明"], "缺少原始来源摘录；不是本轮新证据")
                self.assertEqual(await memory.retrieve(8, "火星网站"), [])
                self.assertTrue(db.memories.revoke(7, 22, 17, expected_version=1))
                self.assertEqual(await memory.retrieve(7, "火星网站"), [])
                self.assertEqual(db.memories.row(7, 17, active=False)["status"], "forgotten")
            finally:
                await memory.close()
                db.close()

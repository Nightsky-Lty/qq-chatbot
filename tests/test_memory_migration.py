import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import Database
from app.memory_schema import _create_records, initialize_memory_schema
from app.memory_retrieval import HybridRetriever
from app.config import Settings
from app.memory import MemoryService


class MemoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bot.db"

    def tearDown(self):
        self.tmp.cleanup()

    def tables(self, conn):
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def legacy(self, *, empty=False):
        conn = sqlite3.connect(self.path)
        conn.executescript("""
          CREATE TABLE memories(id INTEGER PRIMARY KEY AUTOINCREMENT,group_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,source_user_id INTEGER NOT NULL,memory_type TEXT NOT NULL,
            content TEXT NOT NULL,vector TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
        """)
        for i, kind in enumerate(("preference", "event", "episode", "commitment", "image", "explicit", "group_meme"), 1):
            conn.execute("INSERT INTO memories VALUES(?,7,11,11,?,?,'{}',100,200)", (i, kind, f"旧内容{i}"))
        if empty:
            conn.execute("DELETE FROM memories")
        conn.commit()
        return conn

    def test_fresh_database_has_only_new_storage_and_no_old_writes(self):
        db = Database(self.path)
        self.assertIn("memory_records", self.tables(db.connection))
        self.assertNotIn("memories", self.tables(db.connection))
        self.assertIsNone(db.memory_migration_backup)
        for name in ("save_memory", "delete_user_memory", "list_user_memories"):
            self.assertFalse(hasattr(db, name))
        db.close()

    def test_early_schema_converted_and_backup_is_original_and_private(self):
        self.legacy().close()
        db = Database(self.path)
        self.assertNotIn("memories", self.tables(db.connection))
        rows = db.connection.execute("SELECT * FROM memory_records ORDER BY id").fetchall()
        self.assertEqual([r["layer"] for r in rows], ["abstract", "event", "event", "event", "perceptual", "abstract", "abstract"])
        self.assertEqual([r["id"] for r in rows], list(range(1, 8)))
        self.assertEqual([r["content"] for r in rows], [f"旧内容{i}" for i in range(1, 8)])
        self.assertTrue(all(r["occurred_at"] == 100 for r in rows))
        self.assertTrue(all(r["sources_json"] == "[]" for r in rows))
        public = HybridRetriever.public(None, dict(rows[0]), include_sources=True)
        self.assertIn("缺少原始来源", public["来源说明"])
        backup = db.memory_migration_backup
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        with sqlite3.connect(backup) as old:
            self.assertIn("memories", self.tables(old))
            self.assertNotIn("memory_records", self.tables(old))
            self.assertEqual(old.execute("SELECT count(*) FROM memories").fetchone()[0], 7)
        self.assertEqual(db.connection.execute("SELECT count(*) FROM memory_events").fetchone()[0], 3)
        self.assertEqual(db.connection.execute("SELECT count(*) FROM memory_percepts").fetchone()[0], 1)
        db.close()
        reopened = Database(self.path)
        self.assertIsNone(reopened.memory_migration_backup)
        self.assertEqual(len(list(backup.parent.glob("*.db"))), 1)
        self.assertEqual(reopened.connection.execute("SELECT count(*) FROM memory_records").fetchone()[0], 7)
        reopened.close()

    def test_current_schema_preserves_links_versions_blocks_jobs_and_scopes(self):
        db = Database(self.path)
        source = db.log_message(7, 11, "昵称", "我喜欢红茶", message_id="123")
        sources = json.dumps([{"id": source, "user_id": 11, "role": "user", "content": "我喜欢红茶", "time": 100}])
        with db.connection as conn:
            for i, status in enumerate(("active", "candidate", "superseded", "forgotten", "archived"), 1):
                conn.execute("""INSERT INTO memory_records(id,group_id,user_id,source_user_id,memory_type,
                  content,created_at,updated_at,sources_json,layer,version,status,metadata_json)
                  VALUES(?,7,11,11,'preference',?,100,200,?,'abstract',3,?,?)""",
                             (i, f"内容{i}", sources, status, '{"conditions":"晚上"}'))
            conn.execute("INSERT INTO memory_revisions VALUES(1,2,'{}',100)")
            conn.execute("INSERT INTO memory_links VALUES(7,1,2,'derived_from')")
            conn.execute("INSERT INTO memory_media_links VALUES(7,1,12)")
            conn.execute("INSERT INTO memory_blocks VALUES(7,11,'key','content','[999]',100)")
            conn.execute("INSERT INTO memory_consolidation_jobs(task_id,namespace,source_ids_json,new_ids_json,due_at,created_at) VALUES('pending',7,'[1]','[1]',0,100)")
            conn.execute("UPDATE sqlite_sequence SET seq=999 WHERE name='memory_records'")
            # 模拟上一版分层表和触发器；rename使旧视图/触发器指向memories。
            conn.execute("ALTER TABLE memory_records RENAME TO memories")
        db.close()
        migrated = Database(self.path)
        conn = migrated.connection
        self.assertEqual([r[0] for r in conn.execute("SELECT status FROM memory_records ORDER BY id")],
                         ["active", "candidate", "superseded", "forgotten", "archived"])
        self.assertEqual(conn.execute("SELECT sources_json FROM memory_records WHERE id=1").fetchone()[0], sources)
        self.assertEqual(conn.execute("SELECT version FROM memory_records WHERE id=1").fetchone()[0], 3)
        for table in ("memory_links", "memory_media_links", "memory_revisions", "memory_blocks", "memory_consolidation_jobs"):
            self.assertEqual(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT content FROM messages WHERE id=?", (source,)).fetchone()[0], "我喜欢红茶")
        self.assertIsNone(migrated.memories.row(8, 1))
        self.assertIsNone(migrated.memories.row(-11, 1))
        self.assertIsNone(migrated.memories.row(7, 4))
        self.assertIn(999, migrated.memories.blocked_sources(7))
        self.assertEqual(conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0], 5)
        with conn:
            cur = conn.execute("INSERT INTO memory_records(group_id,user_id,source_user_id,memory_type,content,created_at,updated_at) VALUES(7,11,11,'event','新事件',100,200)")
        self.assertEqual(cur.lastrowid, 1000)
        self.assertTrue(conn.execute("SELECT 1 FROM memory_index_jobs WHERE memory_id=1000").fetchone())
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE sql LIKE '% ON \"memories\"%'").fetchone())
        migrated.close()

    def test_deleted_id_high_watermark_retained_when_empty(self):
        conn = self.legacy(empty=True)
        initialize_memory_schema(conn, self.path)
        cur = conn.execute("INSERT INTO memory_records(group_id,user_id,source_user_id,memory_type,content,created_at,updated_at) VALUES(7,11,11,'event','新内容',1,1)")
        self.assertEqual(cur.lastrowid, 8)
        conn.close()

    def test_backup_includes_committed_wal(self):
        writer = self.legacy()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO memories VALUES(8,-11,11,11,'preference','WAL中的偏好','{}',100,200)")
        writer.commit()
        self.assertTrue(Path(str(self.path) + "-wal").exists())
        db = Database(self.path)
        with sqlite3.connect(db.memory_migration_backup) as backup:
            self.assertEqual(backup.execute("SELECT content FROM memories WHERE id=8").fetchone()[0], "WAL中的偏好")
        self.assertEqual(db.memories.row(-11, 8)["content"], "WAL中的偏好")
        db.close()
        writer.close()

    def test_invalid_json_rolls_back_and_retains_legacy(self):
        conn = self.legacy()
        conn.execute("ALTER TABLE memories ADD COLUMN sources_json TEXT DEFAULT 'bad json'")
        conn.commit()
        with self.assertRaises(ValueError):
            initialize_memory_schema(conn, self.path)
        self.assertIn("memories", self.tables(conn))
        self.assertNotIn("memory_records", self.tables(conn))
        self.assertEqual(conn.execute("SELECT count(*) FROM memories").fetchone()[0], 7)
        conn.close()

    def test_copy_field_mismatch_aborts_before_drop(self):
        conn = self.legacy()

        def corrupt_copy(connection):
            _create_records(connection)
            connection.execute("CREATE TRIGGER corrupt AFTER INSERT ON memory_records BEGIN UPDATE memory_records SET updated_at=0 WHERE id=new.id; END")

        with patch("app.memory_schema._create_records", side_effect=corrupt_copy):
            with self.assertRaisesRegex(RuntimeError, "逐字段"):
                initialize_memory_schema(conn, self.path)
        self.assertIn("memories", self.tables(conn))
        self.assertNotIn("memory_records", self.tables(conn))
        conn.close()

    def test_backup_failure_never_changes_old_schema(self):
        conn = self.legacy()
        with patch("app.memory_schema._backup", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                initialize_memory_schema(conn, self.path)
        self.assertIn("memories", self.tables(conn))
        self.assertNotIn("memory_records", self.tables(conn))
        conn.close()

    def test_conflicting_tables_and_unknown_columns_fail_closed(self):
        conn = self.legacy()
        _create_records(conn)
        with self.assertRaisesRegex(RuntimeError, "同时存在"):
            initialize_memory_schema(conn, self.path)
        conn.execute("DROP TABLE memory_records")
        conn.execute("ALTER TABLE memories ADD COLUMN unknown TEXT")
        conn.commit()
        with self.assertRaisesRegex(RuntimeError, "未知字段"):
            initialize_memory_schema(conn, self.path)
        self.assertIn("memories", self.tables(conn))
        conn.close()

    def test_cli_read_only_then_apply_and_repeat(self):
        self.legacy().close()
        script = Path(__file__).resolve().parents[1] / "scripts/migrate_memory_storage.py"

        def run(*extra):
            result = subprocess.run([sys.executable, str(script), "--database", str(self.path), *extra],
                                    capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        before = run()
        self.assertEqual(before["storage"], "memories")
        self.assertFalse((self.path.parent / "backups").exists())
        after = run("--apply")
        self.assertEqual(after["total"], 7)
        self.assertEqual(after["storage"], "memory_records")
        self.assertFalse(after["legacy_table_present"])
        self.assertTrue(Path(after["backup"]).is_file())
        self.assertIsNone(run("--apply")["backup"])


class MemoryMigrationRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_migrated_memory_is_retrievable_and_can_be_forgotten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bot.db"
            db = Database(path)
            with db.connection as conn:
                conn.execute("""INSERT INTO memory_records(id,group_id,user_id,source_user_id,memory_type,
                    content,created_at,updated_at,layer) VALUES(17,7,22,22,'event','QQ22修复了火星网站连接错误',100,200,'event')""")
                conn.execute("ALTER TABLE memory_records RENAME TO memories")
            db.close()
            db = Database(path)
            cfg = Settings(_env_file=None, embedding_provider="hash", memory_qdrant_url="",
                           memory_source_search_enabled=False)
            memory = MemoryService(cfg, db)
            try:
                found = await memory.retrieve(7, "火星网站连接错误", user_ids={11}, layer="event")
                self.assertEqual(found[0]["记忆编号"], 17)
                self.assertIn("缺少原始来源", found[0]["来源说明"])
                self.assertEqual(await memory.retrieve(8, "火星网站"), [])
                self.assertTrue(db.memories.revoke(7, 22, 17, expected_version=1))
                self.assertEqual(await memory.retrieve(7, "火星网站"), [])
                self.assertEqual(db.memories.row(7, 17, active=False)["status"], "forgotten")
            finally:
                await memory.close()
                db.close()

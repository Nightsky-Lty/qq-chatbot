import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.memory import MemoryService, contains_sensitive_memory, sparse_vector, vector_similarity


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tempdir.name) / "test.db")
        self.settings = Settings(
            memory_enabled=True,
            memory_retrieval_limit=5,
            memory_candidate_limit=50,
            memory_max_per_user=10,
        )
        self.memory = MemoryService(self.settings, self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tempdir.cleanup()

    def test_member_identity_preserves_aliases(self) -> None:
        self.db.upsert_group_member(100, 42, "小明", "member")
        self.db.upsert_group_member(100, 42, "明神", "admin")

        member = self.db.group_members(100, {42})[0]
        self.assertEqual(member["user_id"], 42)
        self.assertEqual(member["display_name"], "明神")
        self.assertEqual(member["aliases"], ["小明"])
        self.assertEqual(member["role"], "admin")

    def test_reply_lookup_and_personal_history_are_group_scoped(self) -> None:
        first = self.db.log_message(100, 42, "小明", "本群消息", message_id="same-id")
        self.db.log_message(100, 43, "老王", "另一位成员", message_id="other")
        self.db.log_message(200, 99, "别群成员", "别群消息", message_id="same-id")

        reply = self.db.message_by_onebot_id(100, "same-id")
        self.assertIsNotNone(reply)
        self.assertEqual((reply["user_id"], reply["content"]), (42, "本群消息"))
        history = self.db.recent_user_messages(100, 42, 10, before_id=first + 10)
        self.assertEqual([row["content"] for row in history], ["本群消息"])

    def test_retrieval_is_scoped_to_group_and_related_users(self) -> None:
        self.memory.remember(
            100, 42, "preference", "喜欢玩Apex", confidence=.95, importance=.8,
            source_message_id="m1",
        )
        self.memory.remember(
            100, 43, "preference", "喜欢玩原神", confidence=.95, importance=.8,
            source_message_id="m2",
        )
        self.memory.remember(
            200, 42, "preference", "喜欢打篮球", confidence=.95, importance=.8,
            source_message_id="m3",
        )

        results = self.memory.retrieve(100, "Apex上分", user_ids={42})
        self.assertEqual([item["内容"] for item in results], ["喜欢玩Apex"])

    def test_memory_is_idempotent_and_owner_can_delete_it(self) -> None:
        first, created = self.memory.remember(
            100, 42, "explicit", "叫我明神", confidence=1, importance=.9,
            source_message_id="m1", explicit=True,
        )
        second, created_again = self.memory.remember(
            100, 42, "explicit", "叫我明神", confidence=1, importance=.9,
            source_message_id="m2", explicit=True,
        )
        self.assertEqual(first, second)
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertFalse(self.db.delete_user_memory(100, 99, first))
        self.assertTrue(self.db.delete_user_memory(100, 42, first))

    def test_unrelated_event_is_not_injected_but_stable_profile_is(self) -> None:
        self.memory.remember(
            100, 42, "event", "上周赢了学校篮球赛", confidence=.95, importance=.8,
            source_message_id="m1",
        )
        self.memory.remember(
            100, 42, "profile", "计算机专业学生", confidence=.95, importance=.8,
            source_message_id="m2",
        )

        results = self.memory.retrieve(100, "今晚吃什么", user_ids={42})
        self.assertEqual([item["内容"] for item in results], ["计算机专业学生"])
        empty_query_results = self.memory.retrieve(100, "", user_ids={42})
        self.assertEqual([item["内容"] for item in empty_query_results], ["计算机专业学生"])

    def test_delete_group_user_data_removes_profile_and_sourced_group_memory(self) -> None:
        self.db.upsert_group_member(100, 42, "小明")
        self.db.log_message(100, 42, "小明", "牢王是本群的梗", message_id="m1")
        self.memory.remember(
            100, 0, "group_meme", "牢王是本群的梗", source_user_id=42,
            confidence=.95, importance=.8, source_message_id="m1",
        )

        messages, memories = self.db.delete_group_user_data(100, 42)
        self.assertEqual((messages, memories), (1, 1))
        self.assertEqual(self.db.group_members(100, {42}), [])
        self.assertEqual(self.db.memory_candidates(100, {42}, 10), [])

    def test_sparse_similarity_and_sensitive_filter(self) -> None:
        related = vector_similarity(sparse_vector("喜欢玩Apex"), sparse_vector("Apex上分"))
        unrelated = vector_similarity(sparse_vector("喜欢玩Apex"), sparse_vector("明天吃火锅"))
        self.assertGreater(related, unrelated)
        self.assertTrue(contains_sensitive_memory("我的验证码：123456"))
        self.assertTrue(contains_sensitive_memory("电话是13800138000"))
        self.assertFalse(contains_sensitive_memory("我喜欢玩Apex"))


if __name__ == "__main__":
    unittest.main()

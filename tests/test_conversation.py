import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from app.agent import AgentOutcome
from app.config import Settings
from app.conversation import ConversationState, GroupConversations, GroupMessage


class FollowupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = Settings(_env_file="config.env", chat_batch_seconds=0, ai_cooldown_seconds=0)
        self.sender = AsyncMock()
        self.runner = AsyncMock(return_value=AgentOutcome(1, 1, 1, wait_for_messages=True))
        self.groups = GroupConversations(self.settings, self.runner)
        self.counter = 0

    async def asyncTearDown(self):
        await self.groups.close()

    def message(self, text="新消息", *, direct=False, user=11):
        self.counter += 1
        return GroupMessage(self.counter, str(self.counter), user, f"用户{user}", text, 90, direct=direct)

    async def run_batch(self, message):
        result = await self.groups.submit(7, message, self.sender)
        state = self.groups.states.get(7)
        if state is not None and state.worker is not None:
            await asyncio.wait_for(state.worker, 2)
        return result

    async def test_config_zero_blocks_idle_random_join_for_any_participant_count(self):
        self.assertTrue(self.settings.random_chat_enabled)
        self.assertEqual((self.settings.random_chat_min_probability,
                          self.settings.random_chat_probability_per_user,
                          self.settings.random_chat_max_probability), (0, 0, 0))
        with patch("app.conversation.random.random", return_value=0):
            for user in range(1, 101):
                self.assertEqual(self.groups.random_probability(user), 0)
                self.assertEqual(await self.run_batch(self.message(user=user)), "ignored")
        self.runner.assert_not_awaited()
        self.assertNotIn(7, self.groups.states)

    async def test_direct_wake_then_unmentioned_new_subject_and_finish(self):
        self.assertEqual(await self.run_batch(self.message("今晚打游戏吗", direct=True)), "accepted")
        state = self.groups.states[7]
        self.assertTrue(state.active(time.monotonic()))
        hard_until = state.hard_until
        self.assertFalse(hasattr(state, "topic"))
        self.assertEqual(await self.run_batch(self.message("换个问题，向量是什么", user=22)), "accepted")
        self.assertEqual(self.runner.await_count, 2)
        batch = self.runner.await_args.args[1]
        self.assertFalse(batch.wakes_followup)
        self.assertFalse(hasattr(batch, "topic"))
        self.assertEqual(batch.focus.text, "换个问题，向量是什么")
        self.assertEqual(state.hard_until, hard_until)
        self.runner.return_value = AgentOutcome(1, 1, 1)
        await self.run_batch(self.message("不聊了"))
        self.assertFalse(state.active(time.monotonic()))
        self.assertEqual(await self.run_batch(self.message()), "ignored")

    async def test_silent_wait_cannot_start_or_refresh_followup(self):
        self.runner.return_value = AgentOutcome(1, 1, 0, wait_for_messages=True)
        await self.run_batch(self.message(direct=True))
        self.assertFalse(self.groups.states[7].active(time.monotonic()))
        self.runner.return_value = AgentOutcome(1, 1, 1, wait_for_messages=True)
        await self.run_batch(self.message(direct=True))
        state = self.groups.states[7]
        deadlines = (state.idle_until, state.hard_until)
        self.runner.return_value = AgentOutcome(1, 1, 0, wait_for_messages=True)
        await self.run_batch(self.message("群友之间的交流", user=22))
        self.assertTrue(state.active(time.monotonic()))
        self.assertEqual((state.idle_until, state.hard_until), deadlines)

    async def test_deadlines_and_new_direct_wake(self):
        await self.run_batch(self.message(direct=True))
        for deadline in ("idle_until", "hard_until"):
            state = self.groups.states[7]
            setattr(state, deadline, time.monotonic() - 1)
            self.assertEqual(await self.run_batch(self.message()), "ignored")
            self.assertFalse(state.active(time.monotonic()))
            await self.run_batch(self.message(direct=True))
            self.assertTrue(self.groups.states[7].active(time.monotonic()))

    async def test_expiry_during_silent_or_speaking_followup_cannot_reopen(self):
        for sent in (0, 1):
            self.runner.return_value = AgentOutcome(1, 1, 1, wait_for_messages=True)
            await self.run_batch(self.message(direct=True))
            state = self.groups.states[7]
            async def expire(*args):
                state.hard_until = time.monotonic() - 1
                return AgentOutcome(1, 1, sent, wait_for_messages=True)
            self.runner.side_effect = expire
            await self.run_batch(self.message())
            self.assertFalse(state.active(time.monotonic()))
            self.runner.side_effect = None

    async def test_error_and_cancel_end_followup(self):
        await self.run_batch(self.message(direct=True))
        state = self.groups.states[7]
        self.runner.return_value = AgentOutcome(1, 1, 1, error="failed", wait_for_messages=True)
        await self.run_batch(self.message())
        self.assertFalse(state.active(time.monotonic()))
        self.runner.return_value = AgentOutcome(1, 1, 1, wait_for_messages=True)
        await self.run_batch(self.message(direct=True))
        state = self.groups.states[7]
        await self.groups.cancel(group_id=7)
        self.assertFalse(state.active(time.monotonic()))
        self.assertNotIn(7, self.groups.states)

    async def test_nonzero_config_can_restore_original_random_mechanism(self):
        self.groups.settings = self.settings.model_copy(update={"random_chat_min_probability": 0.01,
            "random_chat_probability_per_user": 0.02, "random_chat_max_probability": 0.20})
        self.assertAlmostEqual(self.groups.random_probability(3), 0.05)
        self.assertAlmostEqual(self.groups.random_probability(100), 0.20)
        with patch("app.conversation.random.random", return_value=0):
            self.assertEqual(await self.run_batch(self.message()), "accepted")
        batch = self.runner.await_args.args[1]
        self.assertTrue(batch.random_trigger)
        self.assertTrue(batch.wakes_followup)
        self.assertTrue(self.groups.states[7].active(time.monotonic()))


class StateTests(unittest.TestCase):
    def test_inactive_state_never_activates_from_deadlines_alone(self):
        state = ConversationState(AsyncMock(), idle_until=100, hard_until=200)
        self.assertFalse(state.active(1))
        self.assertEqual((state.idle_until, state.hard_until), (0, 0))

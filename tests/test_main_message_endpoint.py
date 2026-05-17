import tempfile
import unittest
from pathlib import Path
from unittest import mock

with mock.patch("quizmaker.gemma.load_model", return_value=(None, None)):
    from fastapi.testclient import TestClient

    from app import main as app_module

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ, Overview
from quizmaker.storage import QuizStore


class _FakeGen:
    def __init__(self, safety_result=True, intent="chat", chat_reply="Reply", suggestions=None):
        self.safety_result = safety_result
        self.intent = intent
        self.chat_reply_text = chat_reply
        self.suggestions_list = suggestions or ["Topic A", "Topic B"]

    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(self, topic, overview, count=3, avoid_questions=None):
        return [
            MCQ(
                question=f"Question {i} about {topic}?",
                choices=["A", "B", "C", "D"],
                answer_indices=[0],
                rationale="R",
            )
            for i in range(count)
        ]

    def check_input_safety(self, user_text):
        return self.safety_result

    def classify_intent(self, user_text, topic, overview_json):
        return self.intent

    def generate_chat_reply(self, topic, overview_json, history, user_text):
        return self.chat_reply_text

    def suggest_topics(self, topic, overview_json, history=None, count=4):
        return self.suggestions_list


class MessageEndpointTests(unittest.TestCase):
    def _setup(self, **gen_kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = QuizStore(Path(tmp.name) / "test.sqlite3")
        self.addCleanup(store.close)
        gen = _FakeGen(**gen_kwargs)
        loop = CoreLoop(store, gen, review_every=99)
        app_module._store = store
        app_module._loop = loop
        client = TestClient(app_module.app)
        conv_id = store.create_conversation()
        return client, store, conv_id

    def test_returns_422_on_empty_text(self):
        client, _, conv_id = self._setup()
        response = client.post(f"/conversations/{conv_id}/message", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_returns_422_on_whitespace_only_text(self):
        client, _, conv_id = self._setup()
        response = client.post(f"/conversations/{conv_id}/message", json={"text": "   "})
        self.assertEqual(response.status_code, 422)

    def test_returns_404_on_missing_conversation(self):
        client, _, _ = self._setup()
        response = client.post("/conversations/999999/message", json={"text": "hello"})
        self.assertEqual(response.status_code, 404)

    def test_blocked_branch_returns_reply_with_refusal_no_other_fields(self):
        client, _, conv_id = self._setup(safety_result=False)
        response = client.post(f"/conversations/{conv_id}/message", json={"text": "unsafe"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cannot help", (data.get("reply") or "").lower())
        self.assertIsNone(data.get("overview"))
        self.assertIsNone(data.get("questions"))
        self.assertIsNone(data.get("suggestions"))
        self.assertIsNone(data.get("review"))

    def test_chat_branch_returns_reply_field(self):
        client, store, conv_id = self._setup(intent="chat", chat_reply="Hello there!")
        store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

        response = client.post(f"/conversations/{conv_id}/message", json={"text": "why?"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("reply"), "Hello there!")
        self.assertIsNone(data.get("overview"))
        self.assertIsNone(data.get("questions"))
        self.assertIsNone(data.get("suggestions"))

    def test_start_topic_branch_returns_overview_and_questions(self):
        client, _, conv_id = self._setup(intent="start_topic")

        response = client.post(
            f"/conversations/{conv_id}/message", json={"text": "photosynthesis"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNotNone(data.get("overview"))
        self.assertIsNotNone(data.get("questions"))
        self.assertGreater(len(data["questions"]), 0)
        self.assertIsNone(data.get("reply"))
        self.assertIsNone(data.get("suggestions"))

    def test_suggest_topics_branch_returns_suggestions_field(self):
        client, store, conv_id = self._setup(
            intent="suggest_topics", suggestions=["Topic X", "Topic Y"]
        )
        store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

        response = client.post(
            f"/conversations/{conv_id}/message", json={"text": "what else?"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("suggestions"), ["Topic X", "Topic Y"])
        self.assertIsNone(data.get("reply"))
        self.assertIsNone(data.get("overview"))
        self.assertIsNone(data.get("questions"))

    def test_more_questions_branch_returns_questions_field_only(self):
        client, store, conv_id = self._setup(intent="more_questions")
        store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

        response = client.post(
            f"/conversations/{conv_id}/message", json={"text": "more please"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNotNone(data.get("questions"))
        self.assertGreater(len(data["questions"]), 0)
        self.assertIsNone(data.get("overview"))
        self.assertIsNone(data.get("reply"))
        self.assertIsNone(data.get("suggestions"))


if __name__ == "__main__":
    unittest.main()

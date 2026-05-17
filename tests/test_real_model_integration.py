import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.gemma import GemmaQuizGenerator, load_model
from quizmaker.storage import QuizStore


RUN_REAL_MODEL = os.getenv("RUN_REAL_MODEL_TEST") == "1"


@unittest.skipUnless(RUN_REAL_MODEL, "set RUN_REAL_MODEL_TEST=1 to load Gemma")
class RealModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.processor = load_model()

    def test_real_model_conversation_quiz_review_and_chat_flow(self):
        started_at = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "real-model.sqlite3")
            try:
                loop = CoreLoop(
                    store,
                    GemmaQuizGenerator(self.model, self.processor),
                    review_every=3,
                )
                conversation_id = store.create_conversation()

                overview, questions = loop.start_topic(
                    conversation_id, "photosynthesis", quiz_count=1
                )

                self.assertGreaterEqual(len(overview.points), 1)
                self.assertEqual(len(questions), 1)
                question = questions[0]
                self.assertEqual(question.topic, "photosynthesis")
                self.assertEqual(len(question.mcq.choices), 4)

                wrong_choice = next(
                    index for index in range(4)
                    if index not in question.mcq.answer_indices
                )
                answer = loop.answer_item(
                    conversation_id, question.item_id, wrong_choice
                )

                self.assertFalse(answer.is_correct)
                self.assertIsNone(answer.review)

                review = loop.next_turn(conversation_id)
                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                self.assertEqual(review.topic, "photosynthesis")

                reply, chat_review = loop.chat(
                    conversation_id, "Explain the key idea in one sentence."
                )
                self.assertIsInstance(reply, str)
                self.assertTrue(reply.strip())
                self.assertIsNone(chat_review)

                messages = store.get_messages(conversation_id)
                question_message = next(
                    message for message in messages if message["kind"] == "question"
                )
                question_content = json.loads(question_message["content_json"])
                self.assertNotIn("answer_index", question_content)
                self.assertNotIn("answer_indices", question_content)
                self.assertNotIn("rationale", question_content)

                answer_message = next(
                    message for message in messages if message["kind"] == "answer"
                )
                answer_content = json.loads(answer_message["content_json"])
                self.assertIn("correct_index", answer_content)
                self.assertIn("correct_indices", answer_content)
                self.assertIn("rationale", answer_content)

                log_count = store.conn.execute(
                    "SELECT COUNT(*) FROM logs WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
                self.assertGreater(log_count, 0)
            finally:
                store.close()

        elapsed = time.monotonic() - started_at
        print(f"[real-model integration] completed in {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()

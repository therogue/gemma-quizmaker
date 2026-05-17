import tempfile
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ, Overview
from quizmaker.storage import QuizStore


class FakeGenerator:
    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(self, topic, overview, count=3):
        return [
            MCQ(
                question=f"Question {idx} about {topic}?",
                choices=["A", "B", "C", "D"],
                answer_index=idx % 4,
                rationale=f"Rationale {idx}",
            )
            for idx in range(count)
        ]


class CoreLoopTests(unittest.TestCase):
    def test_rejects_invalid_configuration_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                with self.assertRaises(ValueError):
                    CoreLoop(store, FakeGenerator(), review_every=0)

                loop = CoreLoop(store, FakeGenerator(), review_every=2)
                conv_id = store.create_conversation()

                with self.assertRaises(ValueError):
                    loop.start_topic(conv_id, "   ")

                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)
                with self.assertRaises(ValueError):
                    loop.answer(conv_id, questions[0], 4)
            finally:
                store.close()

    def test_wrong_answer_reappears_as_review_after_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=2)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                is_correct, _ = loop.answer(conv_id, questions[0], 1)
                self.assertFalse(is_correct)
                self.assertIsNone(loop.next_turn(conv_id))

                review = loop.next_turn(conv_id)
                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                self.assertEqual(review.mcq.question, questions[0].mcq.question)
            finally:
                store.close()

    def test_wrong_answer_is_not_due_until_cooldown_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                loop.answer(conv_id, questions[0], 1)
                self.assertIsNone(store.due_review_item(conv_id))

                review = loop.next_turn(conv_id)
                self.assertIsNotNone(review)
                self.assertEqual(review.item_id, questions[0].item_id)
            finally:
                store.close()

    def test_correct_review_demotes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "atoms", quiz_count=1)
                loop.answer(conv_id, questions[0], 1)

                review = loop.next_turn(conv_id)
                self.assertIsNotNone(review)
                is_correct, _ = loop.answer(conv_id, review, review.mcq.answer_index)
                self.assertTrue(is_correct)
                self.assertIsNone(store.due_review_item(conv_id))
            finally:
                store.close()

    def test_answer_item_returns_frontend_result_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "atoms", quiz_count=1)

                result = loop.answer_item(conv_id, questions[0].item_id, 1)

                self.assertEqual(result.item_id, questions[0].item_id)
                self.assertFalse(result.is_correct)
                self.assertEqual(result.correct_index, 0)
                self.assertEqual(result.rationale, "Rationale 0")
                self.assertIsNotNone(store.get_quiz_item(questions[0].item_id, conv_id))
            finally:
                store.close()

    def test_answer_item_rejects_unknown_item_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                with self.assertRaises(ValueError):
                    loop.answer_item(conv_id, 999, 0)
            finally:
                store.close()

    def test_session_and_review_queue_resume_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "quiz.sqlite3"
            store = QuizStore(db_path)
            conv_id = None
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "atoms", quiz_count=1)
                loop.answer(conv_id, questions[0], 1)
            finally:
                store.close()

            reopened_store = QuizStore(db_path)
            try:
                resumed_loop = CoreLoop(reopened_store, FakeGenerator(), review_every=1)
                conv = reopened_store.get_conversation(conv_id)
                self.assertEqual(conv["topic"], "atoms")
                self.assertIn("Overview for atoms", conv["overview_json"])

                review = resumed_loop.next_turn(conv_id)
                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                self.assertEqual(review.mcq.question, questions[0].mcq.question)
            finally:
                reopened_store.close()

    def test_review_queue_returns_highest_priority_due_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                conv_id = store.create_conversation()
                low_id = store.add_quiz_item(
                    conv_id,
                    "topic",
                    MCQ("Low priority?", ["A", "B", "C", "D"], 0, "Because A"),
                    priority=1,
                )
                high_id = store.add_quiz_item(
                    conv_id,
                    "topic",
                    MCQ("High priority?", ["A", "B", "C", "D"], 1, "Because B"),
                    priority=5,
                )

                review = store.due_review_item(conv_id)
                self.assertIsNotNone(review)
                self.assertEqual(review.id, high_id)
                self.assertNotEqual(review.id, low_id)
            finally:
                store.close()

    def test_two_conversations_are_fully_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_a = store.create_conversation()
                conv_b = store.create_conversation()

                _, questions_a = loop.start_topic(conv_a, "cells", quiz_count=1)
                _, questions_b = loop.start_topic(conv_b, "atoms", quiz_count=1)

                # Answer wrong in conv_a, then advance turn to clear cooldown
                loop.answer(conv_a, questions_a[0], 1)
                review_a = loop.next_turn(conv_a)  # decrements cooldown, injects review

                # conv_a gets a review; conv_b review queue stays empty
                self.assertIsNotNone(review_a)
                self.assertIsNone(store.due_review_item(conv_b))

                # Items from conv_a must not be accessible via conv_b
                self.assertIsNone(store.get_quiz_item(questions_a[0].item_id, conv_b))
                self.assertIsNotNone(store.get_quiz_item(questions_a[0].item_id, conv_a))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ
from quizmaker.storage import QuizStore


class FakeGenerator:
    def generate_overview(self, topic):
        return f"Overview for {topic}"

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
                with self.assertRaises(ValueError):
                    loop.start_topic("   ")

                _, questions = loop.start_topic("cells", quiz_count=1)
                with self.assertRaises(ValueError):
                    loop.answer(questions[0], 4)
            finally:
                store.close()

    def test_wrong_answer_reappears_as_review_after_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=2)
                _, questions = loop.start_topic("cells", quiz_count=1)

                is_correct, _ = loop.answer(questions[0], 1)
                self.assertFalse(is_correct)
                self.assertIsNone(loop.next_turn())

                review = loop.next_turn()
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
                _, questions = loop.start_topic("cells", quiz_count=1)

                loop.answer(questions[0], 1)
                self.assertIsNone(store.due_review_item())

                review = loop.next_turn()
                self.assertIsNotNone(review)
                self.assertEqual(review.item_id, questions[0].item_id)
            finally:
                store.close()

    def test_correct_review_demotes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                _, questions = loop.start_topic("atoms", quiz_count=1)
                loop.answer(questions[0], 1)

                review = loop.next_turn()
                self.assertIsNotNone(review)
                is_correct, _ = loop.answer(review, review.mcq.answer_index)
                self.assertTrue(is_correct)
                self.assertIsNone(store.due_review_item())
            finally:
                store.close()

    def test_session_and_review_queue_resume_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "quiz.sqlite3"
            store = QuizStore(db_path)
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                _, questions = loop.start_topic("atoms", quiz_count=1)
                loop.answer(questions[0], 1)
            finally:
                store.close()

            reopened_store = QuizStore(db_path)
            try:
                resumed_loop = CoreLoop(reopened_store, FakeGenerator(), review_every=1)
                self.assertEqual(resumed_loop.topic, "atoms")
                self.assertEqual(resumed_loop.overview, "Overview for atoms")

                review = resumed_loop.next_turn()
                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                self.assertEqual(review.mcq.question, questions[0].mcq.question)
            finally:
                reopened_store.close()

    def test_review_queue_returns_highest_priority_due_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                low_id = store.add_quiz_item(
                    "topic",
                    MCQ("Low priority?", ["A", "B", "C", "D"], 0, "Because A"),
                    priority=1,
                )
                high_id = store.add_quiz_item(
                    "topic",
                    MCQ("High priority?", ["A", "B", "C", "D"], 1, "Because B"),
                    priority=5,
                )

                review = store.due_review_item()
                self.assertIsNotNone(review)
                self.assertEqual(review.id, high_id)
                self.assertNotEqual(review.id, low_id)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

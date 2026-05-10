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


if __name__ == "__main__":
    unittest.main()

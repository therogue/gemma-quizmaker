import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ
from quizmaker.storage import QuizStore


EXPECTED_M2_NODES = {
    "overview",
    "quiz_gen",
    "verify",
    "safety",
    "grade",
    "review_inject",
}


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


class SequenceGenerator:
    def __init__(self, mcqs):
        self.mcqs = list(mcqs)
        self.quiz_calls = 0

    def generate_overview(self, topic):
        return f"Overview for {topic}"

    def generate_quiz(self, topic, overview, count=3):
        self.quiz_calls += 1
        start = self.quiz_calls - 1
        return self.mcqs[start : start + count]


class QuestionVerifier:
    def __init__(self, accepted_question):
        self.accepted_question = accepted_question
        self.calls = []

    def verify_mcq(self, topic, overview, mcq):
        self.calls.append(mcq.question)
        return mcq.question == self.accepted_question


class AlwaysRejectVerifier:
    def __init__(self):
        self.calls = 0

    def verify_mcq(self, topic, overview, mcq):
        self.calls += 1
        return False


class RejectingSafetyChecker:
    def is_safe(self, topic, overview, mcqs):
        return False


def read_log_nodes(log_path):
    return [entry["node"] for entry in read_log_entries(log_path)]


def read_log_entries(log_path):
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class LangGraphCoreLoopStructureTests(unittest.TestCase):
    def test_project_declares_langgraph_dependency(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as handle:
            config = tomllib.load(handle)

        dependencies = config["project"]["dependencies"]
        self.assertTrue(
            any(dependency.lower().startswith("langgraph") for dependency in dependencies),
            "M2 core loop should declare LangGraph as a runtime dependency",
        )

    def test_core_loop_exposes_langgraph_runtime_and_m2_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                try:
                    loop = CoreLoop(
                        store,
                        FakeGenerator(),
                        review_every=2,
                        log_path=Path(tmp) / "nodes.jsonl",
                    )
                except TypeError as exc:
                    self.fail(f"CoreLoop does not expose M2 graph/logging options: {exc}")

                self.assertTrue(hasattr(loop, "graph"))
                self.assertTrue(hasattr(loop.graph, "invoke"))
                self.assertEqual(set(loop.graph_node_names), EXPECTED_M2_NODES)
            finally:
                store.close()


class LangGraphCoreLoopBehaviorTests(unittest.TestCase):
    def test_start_topic_runs_graph_nodes_and_logs_node_inputs_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(
                    store,
                    FakeGenerator(),
                    review_every=2,
                    log_path=log_path,
                )

                overview, questions = loop.start_topic("cells", quiz_count=1)

                self.assertEqual(overview, "Overview for cells")
                self.assertEqual(len(questions), 1)
                self.assertEqual(
                    read_log_nodes(log_path),
                    ["overview", "quiz_gen", "verify", "safety"],
                )
                for entry in read_log_entries(log_path):
                    with self.subTest(node=entry["node"]):
                        self.assertIsInstance(entry["input"], dict)
                        self.assertIsInstance(entry["output"], dict)
                        self.assertTrue(entry["input"])
                        self.assertTrue(entry["output"])
            finally:
                store.close()

    def test_verify_failure_retries_once_then_accepts_replacement_item(self):
        bad_mcq = MCQ("Bad question?", ["A", "B", "C", "D"], 0, "Bad rationale")
        good_mcq = MCQ("Good question?", ["A", "B", "C", "D"], 1, "Good rationale")
        generator = SequenceGenerator([bad_mcq, good_mcq])
        verifier = QuestionVerifier("Good question?")

        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                try:
                    loop = CoreLoop(store, generator, verifier=verifier)
                except TypeError as exc:
                    self.fail(f"CoreLoop does not accept verifier dependency: {exc}")

                _, questions = loop.start_topic("cells", quiz_count=1)

                self.assertEqual([question.mcq.question for question in questions], ["Good question?"])
                self.assertEqual(generator.quiz_calls, 2)
                self.assertEqual(verifier.calls, ["Bad question?", "Good question?"])
            finally:
                store.close()

    def test_verify_failure_drops_item_after_one_retry(self):
        bad_mcq = MCQ("Bad question?", ["A", "B", "C", "D"], 0, "Bad rationale")
        second_bad_mcq = MCQ(
            "Still bad question?",
            ["A", "B", "C", "D"],
            1,
            "Still bad rationale",
        )
        generator = SequenceGenerator([bad_mcq, second_bad_mcq])
        verifier = AlwaysRejectVerifier()

        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                try:
                    loop = CoreLoop(store, generator, verifier=verifier)
                except TypeError as exc:
                    self.fail(f"CoreLoop does not accept verifier dependency: {exc}")

                _, questions = loop.start_topic("cells", quiz_count=1)

                self.assertEqual(questions, [])
                self.assertEqual(generator.quiz_calls, 2)
                self.assertEqual(verifier.calls, 2)
            finally:
                store.close()

    def test_safety_failure_refuses_before_content_is_delivered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                try:
                    loop = CoreLoop(store, FakeGenerator(), safety_checker=RejectingSafetyChecker())
                except TypeError as exc:
                    self.fail(f"CoreLoop does not accept safety checker dependency: {exc}")

                overview, questions = loop.start_topic("unsafe topic", quiz_count=1)

                self.assertIn("cannot help", overview.lower())
                self.assertEqual(questions, [])
                self.assertIsNone(store.due_review_item())
            finally:
                store.close()

    def test_review_injection_runs_through_graph_and_logs_review_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(
                    store,
                    FakeGenerator(),
                    review_every=1,
                    log_path=log_path,
                )
                _, questions = loop.start_topic("cells", quiz_count=1)
                loop.answer(questions[0], 1)

                review = loop.next_turn()

                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                review_entries = [
                    entry for entry in read_log_entries(log_path) if entry["node"] == "review_inject"
                ]
                self.assertEqual(len(review_entries), 1)
                self.assertTrue(review_entries[0]["input"])
                self.assertTrue(review_entries[0]["output"])
                self.assertIsNotNone(review_entries[0]["output"]["review"])
            finally:
                store.close()

    def test_answer_grading_runs_through_graph_and_logs_grade_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(
                    store,
                    FakeGenerator(),
                    review_every=1,
                    log_path=log_path,
                )
                _, questions = loop.start_topic("cells", quiz_count=1)

                result = loop.answer(questions[0], 1)

                self.assertFalse(result[0])
                entries = read_log_entries(log_path)
                grade_entries = [entry for entry in entries if entry["node"] == "grade"]
                self.assertEqual(len(grade_entries), 1)
                self.assertEqual(grade_entries[0]["input"]["choice_index"], 1)
                self.assertFalse(grade_entries[0]["output"]["result"]["is_correct"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

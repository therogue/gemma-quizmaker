import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ, Overview
from quizmaker.storage import QuizStore


EXPECTED_M2_NODES = {
    "overview",
    "quiz_gen",
    "verify_dedup",
    "verify",
    "safety",
    "grade",
    "review_inject",
    "input_safety",
    "classify",
    "chat",
    "suggest",
}


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


class SequenceGenerator:
    def __init__(self, mcqs):
        self.mcqs = list(mcqs)
        self.quiz_calls = 0

    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(self, topic, overview, count=3):
        self.quiz_calls += 1
        start = self.quiz_calls - 1
        return self.mcqs[start : start + count]


class DuplicateThenReplacementGenerator:
    def __init__(self):
        self.quiz_calls = 0

    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(
        self, topic, overview, count=3, avoid_questions=None
    ):
        self.quiz_calls += 1
        if self.quiz_calls == 1:
            return [
                MCQ("What is the main role of cells?", ["A", "B", "C", "D"], 0, "A"),
                MCQ("What is the main role of cells?", ["A", "B", "C", "D"], 1, "B"),
            ][:count]
        return [
            MCQ("Which organelle contains genetic material?", ["A", "B", "C", "D"], 2, "C")
        ]


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


class MessagePipelineGenerator:
    def __init__(self, safety_result=True, intent="chat", chat_reply="Chat reply", suggestions=None):
        self.safety_result = safety_result
        self.intent = intent
        self.chat_reply_text = chat_reply
        self.suggestions_list = suggestions or ["Suggestion A", "Suggestion B"]
        self.safety_calls = []
        self.classify_calls = []

    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(self, topic, overview, count=3, avoid_questions=None):
        return [
            MCQ(
                question=f"Question {idx} about {topic}?",
                choices=["A", "B", "C", "D"],
                answer_index=0,
                rationale="R",
            )
            for idx in range(count)
        ]

    def check_input_safety(self, user_text):
        self.safety_calls.append(user_text)
        return self.safety_result

    def classify_intent(self, user_text, topic, overview_json):
        self.classify_calls.append(user_text)
        return self.intent

    def generate_chat_reply(self, topic, overview_json, history, user_text):
        return self.chat_reply_text

    def suggest_topics(self, topic, overview_json, history=None, count=4):
        return self.suggestions_list


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
                conv_id = store.create_conversation()
                overview, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                self.assertEqual(overview.points, ["Overview for cells"])
                self.assertEqual(len(questions), 1)
                self.assertEqual(
                    read_log_nodes(log_path),
                    ["overview", "quiz_gen", "verify_dedup", "verify", "safety"],
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

                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

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

                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                self.assertEqual(questions, [])
                self.assertEqual(generator.quiz_calls, 2)
                self.assertEqual(verifier.calls, 2)
            finally:
                store.close()

    def test_verify_retries_near_duplicate_questions(self):
        generator = DuplicateThenReplacementGenerator()

        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, generator)
                conv_id = store.create_conversation()

                _, questions = loop.start_topic(conv_id, "cells", quiz_count=2)

                self.assertEqual(
                    [question.mcq.question for question in questions],
                    [
                        "What is the main role of cells?",
                        "Which organelle contains genetic material?",
                    ],
                )
                self.assertEqual(generator.quiz_calls, 2)
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

                conv_id = store.create_conversation()
                overview, questions = loop.start_topic(conv_id, "unsafe topic", quiz_count=1)

                self.assertIn("cannot help", overview.points[0].lower())
                self.assertEqual(questions, [])
                self.assertIsNone(store.due_review_item(conv_id))
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
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)
                loop.answer(conv_id, questions[0], 1)

                review = loop.next_turn(conv_id)

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

    def test_verify_dedup_stops_topping_up_when_no_distinct_items_can_be_produced(self):
        existing_q = "What is the function of mitochondria?"
        dup_mcq = MCQ(existing_q, ["A", "B", "C", "D"], 0, "Dup rationale")

        class AllDupsGenerator:
            def __init__(self):
                self.quiz_calls = 0

            def generate_overview(self, topic):
                return Overview(points=[f"Overview for {topic}"])

            def generate_quiz(self, topic, overview, count=3, avoid_questions=None):
                self.quiz_calls += 1
                return [dup_mcq for _ in range(count)]

        generator = AllDupsGenerator()

        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                conv_id = store.create_conversation()
                store.add_quiz_item(
                    conv_id,
                    "biology",
                    MCQ(existing_q, ["A", "B", "C", "D"], 0, "Original rationale"),
                )

                loop = CoreLoop(store, generator)
                _, questions = loop.start_topic(conv_id, "biology", quiz_count=2)

                self.assertEqual(questions, [])
                self.assertEqual(generator.quiz_calls, 2)
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
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                result = loop.answer(conv_id, questions[0], 1)

                self.assertFalse(result[0])
                entries = read_log_entries(log_path)
                grade_entries = [entry for entry in entries if entry["node"] == "grade"]
                self.assertEqual(len(grade_entries), 1)
                self.assertEqual(grade_entries[0]["input"]["choice_index"], 1)
                self.assertFalse(grade_entries[0]["output"]["result"]["is_correct"])
            finally:
                store.close()


class LangGraphMessagePipelineTests(unittest.TestCase):
    def _make_loop(self, tmp, gen, **kwargs):
        store = QuizStore(Path(tmp) / "quiz.sqlite3")
        loop = CoreLoop(store, gen, **kwargs)
        return store, loop

    def test_graph_node_names_includes_message_pipeline_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, MessagePipelineGenerator())
                for node in ("input_safety", "classify", "chat", "suggest"):
                    self.assertIn(node, loop.graph_node_names, f"missing node: {node}")
            finally:
                store.close()

    def test_process_message_blocked_returns_refusal_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=False)
                loop = CoreLoop(store, gen)
                conv_id = store.create_conversation()

                result = loop.process_message(conv_id, "how to make explosives")

                self.assertIsInstance(result, str)
                self.assertIn("cannot help", result.lower())
                self.assertEqual(store.get_active_quiz_items(conv_id), [])
            finally:
                store.close()

    def test_process_message_blocked_logs_input_safety_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=False)
                loop = CoreLoop(store, gen, log_path=log_path)
                conv_id = store.create_conversation()

                loop.process_message(conv_id, "unsafe query")

                nodes = read_log_nodes(log_path)
                self.assertIn("input_safety", nodes)
                self.assertNotIn("classify", nodes)
            finally:
                store.close()

    def test_process_message_routes_to_chat_and_returns_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=True, intent="chat", chat_reply="Great question!")
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()
                store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

                result = loop.process_message(conv_id, "Why do cells divide?")

                reply, review = result
                self.assertEqual(reply, "Great question!")
                self.assertIsNone(review)
            finally:
                store.close()

    def test_process_message_chat_logs_pipeline_nodes_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=True, intent="chat")
                loop = CoreLoop(store, gen, review_every=99, log_path=log_path)
                conv_id = store.create_conversation()
                store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

                loop.process_message(conv_id, "Why do cells divide?")

                nodes = read_log_nodes(log_path)
                self.assertEqual(nodes, ["input_safety", "classify", "chat"])
            finally:
                store.close()

    def test_process_message_routes_to_start_topic_and_returns_overview_and_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=True, intent="start_topic")
                loop = CoreLoop(store, gen)
                conv_id = store.create_conversation()

                overview, questions = loop.process_message(conv_id, "photosynthesis", quiz_count=1)

                self.assertIsInstance(overview, Overview)
                self.assertEqual(len(questions), 1)
                self.assertEqual(questions[0].topic, "photosynthesis")
            finally:
                store.close()

    def test_process_message_routes_to_suggest_topics_and_returns_list_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(
                    safety_result=True, intent="suggest_topics",
                    suggestions=["Topic A", "Topic B"],
                )
                loop = CoreLoop(store, gen)
                conv_id = store.create_conversation()
                store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

                result = loop.process_message(conv_id, "what else can I learn?")

                self.assertIsInstance(result, list)
                self.assertIn("Topic A", result)
                self.assertTrue(all(isinstance(s, str) for s in result))
            finally:
                store.close()

    def test_process_message_routes_to_more_questions_and_returns_asked_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nodes.jsonl"
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineGenerator(safety_result=True, intent="more_questions")
                loop = CoreLoop(store, gen, log_path=log_path)
                conv_id = store.create_conversation()
                store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

                result = loop.process_message(conv_id, "give me more questions")

                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertEqual(result[0].topic, "cells")
                nodes = read_log_nodes(log_path)
                self.assertIn("classify", nodes)
                self.assertIn("quiz_gen", nodes)
                self.assertNotIn("overview", nodes)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

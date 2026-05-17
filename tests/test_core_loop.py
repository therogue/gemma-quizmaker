import json
import tempfile
import unittest
from pathlib import Path

from quizmaker.core_loop import CoreLoop
from quizmaker.schemas import MCQ, Overview
from quizmaker.storage import QuizStore


class FakeGenerator:
    def generate_overview(self, topic):
        return Overview(points=[f"Overview for {topic}"])

    def generate_quiz(self, topic, overview, count=3, avoid_questions=None):
        return [
            MCQ(
                question=f"Question {idx} about {topic}?",
                choices=["A", "B", "C", "D"],
                answer_index=idx % 4,
                rationale=f"Rationale {idx}",
            )
            for idx in range(count)
        ]


class SuggestingGenerator(FakeGenerator):
    def __init__(self):
        self.suggestion_calls = 0

    def suggest_topics(self, topic, overview_json, history=None, count=4):
        self.suggestion_calls += 1
        return ["Suggested topic"]


class MultiAnswerGenerator(FakeGenerator):
    def generate_quiz(self, topic, overview, count=3, avoid_questions=None):
        return [
            MCQ(
                question=f"Which statements about {topic} are true?",
                choices=["First true statement", "False statement", "Second true statement", "Other false statement"],
                answer_indices=[0, 2],
                rationale="The first and third choices are true.",
            )
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

    def test_answer_item_grades_multiple_selected_answers_as_a_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, MultiAnswerGenerator(), review_every=99)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                wrong = loop.answer_item(conv_id, questions[0].item_id, [0])
                self.assertFalse(wrong.is_correct)
                self.assertEqual(wrong.correct_indices, [0, 2])

                answer_message = next(
                    message for message in store.get_messages(conv_id)
                    if message["kind"] == "answer"
                )
                answer_content = json.loads(answer_message["content_json"])
                self.assertEqual(answer_content["choice_indices"], [0])
                self.assertEqual(answer_content["correct_indices"], [0, 2])
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

    def test_start_topic_counts_as_turn_without_auto_suggestions_or_answer_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            generator = SuggestingGenerator()
            try:
                loop = CoreLoop(store, generator, review_every=3)
                conv_id = store.create_conversation()

                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                conv = store.get_conversation(conv_id)
                self.assertEqual(conv["turn_count"], 1)
                self.assertEqual(questions[0].topic, "cells")
                self.assertEqual(generator.suggestion_calls, 0)

                messages = store.get_messages(conv_id)
                kinds = [message["kind"] for message in messages]
                self.assertNotIn("suggestions", kinds)
                question_message = next(message for message in messages if message["kind"] == "question")
                question_content = json.loads(question_message["content_json"])
                self.assertEqual(question_content["topic"], "cells")
                self.assertNotIn("answer_index", question_content)
                self.assertNotIn("answer_indices", question_content)
                self.assertNotIn("rationale", question_content)
            finally:
                store.close()

    def test_answer_counts_as_turn_and_reveals_answer_in_answer_message_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=99)
                conv_id = store.create_conversation()
                _, questions = loop.start_topic(conv_id, "cells", quiz_count=1)

                result = loop.answer_item(conv_id, questions[0].item_id, 1)

                self.assertIsNone(result.review)
                conv = store.get_conversation(conv_id)
                self.assertEqual(conv["turn_count"], 2)

                answer_message = next(
                    message for message in store.get_messages(conv_id)
                    if message["kind"] == "answer"
                )
                answer_content = json.loads(answer_message["content_json"])
                self.assertEqual(answer_content["correct_index"], 0)
                self.assertEqual(answer_content["correct_indices"], [0])
                self.assertEqual(answer_content["rationale"], "Rationale 0")
            finally:
                store.close()

    def test_review_question_carries_source_topic_after_focus_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=1)
                conv_id = store.create_conversation()
                _, cells_questions = loop.start_topic(conv_id, "cells", quiz_count=1)
                loop.answer(conv_id, cells_questions[0], 1)
                loop.start_topic(conv_id, "atoms", quiz_count=1)

                review = loop.next_turn(conv_id)

                self.assertIsNotNone(review)
                self.assertTrue(review.is_review)
                self.assertEqual(review.topic, "cells")
            finally:
                store.close()

    def test_graph_events_are_persisted_to_logs_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, FakeGenerator(), review_every=2)
                conv_id = store.create_conversation()

                loop.start_topic(conv_id, "cells", quiz_count=1)

                rows = store.conn.execute(
                    "SELECT event FROM logs WHERE conversation_id = ? ORDER BY id",
                    (conv_id,),
                ).fetchall()
                self.assertEqual(
                    [row["event"] for row in rows],
                    [
                        "graph.overview",
                        "graph.quiz_gen",
                        "graph.verify_dedup",
                        "graph.verify",
                        "graph.safety",
                    ],
                )
            finally:
                store.close()

    def test_get_question_texts_returns_questions_for_one_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                conv_a = store.create_conversation()
                conv_b = store.create_conversation()
                store.add_quiz_item(
                    conv_a, "biology",
                    MCQ("Question A1?", ["a", "b", "c", "d"], 0, "R"),
                )
                store.add_quiz_item(
                    conv_a, "biology",
                    MCQ("Question A2?", ["a", "b", "c", "d"], 0, "R"),
                )
                store.add_quiz_item(
                    conv_b, "biology",
                    MCQ("Question B1?", ["a", "b", "c", "d"], 0, "R"),
                )

                texts_a = store.get_question_texts(conv_a)

                self.assertEqual(sorted(texts_a), ["Question A1?", "Question A2?"])
                self.assertNotIn("Question B1?", texts_a)
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


class MessagePipelineFakeGenerator(FakeGenerator):
    def __init__(self, safety_result=True, intent="chat", chat_reply="Chat reply", suggestions=None):
        self.safety_result = safety_result
        self.intent = intent
        self.chat_reply_text = chat_reply
        self.suggestions_list = suggestions or ["Topic A"]

    def check_input_safety(self, user_text):
        return self.safety_result

    def classify_intent(self, user_text, topic, overview_json):
        return self.intent

    def generate_chat_reply(self, topic, overview_json, history, user_text):
        return self.chat_reply_text

    def suggest_topics(self, topic, overview_json, history=None, count=4):
        return self.suggestions_list


class MessagePipelineIntegrationTests(unittest.TestCase):
    def test_process_message_chat_branch_persists_user_and_assistant_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineFakeGenerator(safety_result=True, intent="chat", chat_reply="Great question!")
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()
                store.save_conversation(conv_id, "cells", '{"points": ["Overview"]}', turn_count=0)

                reply, review = loop.process_message(conv_id, "Why do cells divide?")

                self.assertEqual(reply, "Great question!")
                self.assertIsNone(review)
                messages = store.get_messages(conv_id)
                chat_messages = [m for m in messages if m["kind"] == "chat"]
                self.assertEqual(len(chat_messages), 2)
                user_msg = next(m for m in chat_messages if m["role"] == "user")
                self.assertIn("Why do cells divide?", json.loads(user_msg["content_json"])["text"])
            finally:
                store.close()

    def test_process_message_start_topic_branch_creates_quiz_items_and_sets_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineFakeGenerator(safety_result=True, intent="start_topic")
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()

                overview, questions = loop.process_message(conv_id, "photosynthesis", quiz_count=1)

                self.assertIsInstance(overview, Overview)
                self.assertEqual(len(questions), 1)
                active = store.get_active_quiz_items(conv_id)
                self.assertEqual(len(active), 1)
                conv = store.get_conversation(conv_id)
                self.assertEqual(conv["topic"], "photosynthesis")
            finally:
                store.close()

    def test_process_message_blocked_stores_no_messages_and_returns_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineFakeGenerator(safety_result=False)
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()

                result = loop.process_message(conv_id, "unsafe query")

                self.assertIsInstance(result, str)
                self.assertIn("cannot help", result.lower())
                self.assertEqual(store.get_messages(conv_id), [])
            finally:
                store.close()

    def test_process_message_raises_value_error_on_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                loop = CoreLoop(store, MessagePipelineFakeGenerator(), review_every=99)
                conv_id = store.create_conversation()

                with self.assertRaises(ValueError):
                    loop.process_message(conv_id, "")
                with self.assertRaises(ValueError):
                    loop.process_message(conv_id, "   ")
            finally:
                store.close()

    def test_process_message_more_questions_intent_falls_back_to_start_topic_when_no_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineFakeGenerator(safety_result=True, intent="more_questions")
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()

                result = loop.process_message(conv_id, "photosynthesis")

                self.assertIsInstance(result, tuple)
                overview, questions = result
                self.assertIsInstance(overview, Overview)
                self.assertGreater(len(questions), 0)
                self.assertEqual(questions[0].topic, "photosynthesis")
            finally:
                store.close()

    def test_process_message_suggest_topics_intent_falls_back_to_start_topic_when_no_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuizStore(Path(tmp) / "quiz.sqlite3")
            try:
                gen = MessagePipelineFakeGenerator(safety_result=True, intent="suggest_topics")
                loop = CoreLoop(store, gen, review_every=99)
                conv_id = store.create_conversation()

                result = loop.process_message(conv_id, "neural networks")

                self.assertIsInstance(result, tuple)
                overview, questions = result
                self.assertIsInstance(overview, Overview)
                self.assertEqual(questions[0].topic, "neural networks")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

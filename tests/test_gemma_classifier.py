import unittest
from unittest import mock

from quizmaker.gemma import GemmaQuizGenerator


class GemmaCheckInputSafetyTests(unittest.TestCase):
    def _gen(self):
        return GemmaQuizGenerator(model=object(), processor=object())

    def test_returns_true_when_model_says_safe(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"safe": true}'):
            self.assertTrue(gen.check_input_safety("Tell me about photosynthesis"))

    def test_returns_false_when_model_says_unsafe(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"safe": false}'):
            self.assertFalse(gen.check_input_safety("How to make explosives"))

    def test_falls_back_to_safe_after_retry_when_both_outputs_malformed(self):
        gen = self._gen()
        with mock.patch(
            "quizmaker.gemma.run_inference", return_value="not json at all"
        ) as mocked:
            self.assertTrue(gen.check_input_safety("some query"))
            self.assertEqual(mocked.call_count, 2)

    def test_retry_returns_recovered_value_when_first_output_malformed(self):
        gen = self._gen()
        with mock.patch(
            "quizmaker.gemma.run_inference",
            side_effect=["garbage", '{"safe": false}'],
        ) as mocked:
            self.assertFalse(gen.check_input_safety("borderline query"))
            self.assertEqual(mocked.call_count, 2)


class GemmaClassifyIntentTests(unittest.TestCase):
    def _gen(self):
        return GemmaQuizGenerator(model=object(), processor=object())

    def test_returns_start_topic_when_model_says_start_topic(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"intent": "start_topic"}'):
            self.assertEqual(gen.classify_intent("photosynthesis", "", ""), "start_topic")

    def test_returns_chat_when_model_says_chat(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"intent": "chat"}'):
            self.assertEqual(gen.classify_intent("Why is chlorophyll green?", "photosynthesis", "{}"), "chat")

    def test_returns_more_questions_when_model_says_more_questions(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"intent": "more_questions"}'):
            self.assertEqual(gen.classify_intent("give me more questions", "photosynthesis", "{}"), "more_questions")

    def test_returns_suggest_topics_when_model_says_suggest_topics(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"intent": "suggest_topics"}'):
            self.assertEqual(gen.classify_intent("what else can I learn?", "photosynthesis", "{}"), "suggest_topics")

    def test_falls_back_to_start_topic_when_no_topic_and_output_invalid(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value="not json"):
            self.assertEqual(gen.classify_intent("something", "", ""), "start_topic")

    def test_falls_back_to_chat_when_topic_exists_and_output_invalid(self):
        gen = self._gen()
        with mock.patch("quizmaker.gemma.run_inference", return_value="not json"):
            self.assertEqual(gen.classify_intent("something", "photosynthesis", "{}"), "chat")


if __name__ == "__main__":
    unittest.main()

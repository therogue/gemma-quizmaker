import unittest
from unittest import mock

from quizmaker.gemma import GemmaQuizVerifier
from quizmaker.schemas import MCQ


class GemmaQuizVerifierTests(unittest.TestCase):
    def _mcq(self, answer_indices):
        return MCQ(
            question="What captures light energy in plants?",
            choices=["Carotenoids", "Chlorophyll", "Cellulose", "Cytosol"],
            answer_indices=answer_indices,
            rationale="Chlorophyll captures light energy.",
        )

    def test_returns_true_when_model_indices_match_answer_indices(self):
        verifier = GemmaQuizVerifier(model=object(), processor=object())
        mcq = self._mcq([1])
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"indices": [1]}'):
            self.assertTrue(verifier.verify_mcq("photosynthesis", "Overview text", mcq))

    def test_returns_false_when_model_indices_disagree(self):
        verifier = GemmaQuizVerifier(model=object(), processor=object())
        mcq = self._mcq([1])
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"indices": [2]}'):
            self.assertFalse(verifier.verify_mcq("photosynthesis", "Overview text", mcq))

    def test_uses_set_equality_for_multi_correct_answers(self):
        verifier = GemmaQuizVerifier(model=object(), processor=object())
        mcq = self._mcq([0, 2])
        with mock.patch("quizmaker.gemma.run_inference", return_value='{"indices": [2, 0]}'):
            self.assertTrue(verifier.verify_mcq("photosynthesis", "Overview text", mcq))

    def test_returns_false_when_model_output_is_malformed_json(self):
        verifier = GemmaQuizVerifier(model=object(), processor=object())
        mcq = self._mcq([1])
        with mock.patch("quizmaker.gemma.run_inference", return_value="not json at all"):
            self.assertFalse(verifier.verify_mcq("photosynthesis", "Overview text", mcq))


if __name__ == "__main__":
    unittest.main()

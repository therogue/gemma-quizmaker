import unittest

from quizmaker.schemas import MCQ


class MCQSchemaTests(unittest.TestCase):
    def valid_data(self):
        return {
            "question": "What is 2 + 2?",
            "choices": ["1", "2", "4", "5"],
            "answer_indices": [2],
            "rationale": "2 + 2 equals 4.",
        }

    def test_accepts_valid_mcq(self):
        mcq = MCQ.from_mapping(self.valid_data())
        self.assertEqual(mcq.question, "What is 2 + 2?")
        self.assertEqual(mcq.answer_index, 2)
        self.assertEqual(mcq.answer_indices, [2])
        self.assertEqual(mcq.choices, ["1", "2", "4", "5"])

    def test_accepts_multiple_answer_indices(self):
        data = self.valid_data()
        data["answer_indices"] = [0, 2]

        mcq = MCQ.from_mapping(data)

        self.assertEqual(mcq.answer_indices, [0, 2])

    def test_rejects_invalid_choice_count(self):
        data = self.valid_data()
        data["choices"] = ["A", "B", "C"]
        with self.assertRaises(ValueError):
            MCQ.from_mapping(data)

    def test_accepts_legacy_answer_index(self):
        data = self.valid_data()
        del data["answer_indices"]
        data["answer_index"] = 2

        mcq = MCQ.from_mapping(data)

        self.assertEqual(mcq.answer_indices, [2])

    def test_rejects_invalid_answer_indices(self):
        for answer_indices in ([], [-1], [4], ["2"], [1, 1], [0, 1, 2, 3]):
            with self.subTest(answer_indices=answer_indices):
                data = self.valid_data()
                data["answer_indices"] = answer_indices
                with self.assertRaises(ValueError):
                    MCQ.from_mapping(data)

    def test_rejects_empty_required_strings(self):
        for field in ("question", "rationale"):
            with self.subTest(field=field):
                data = self.valid_data()
                data[field] = "   "
                with self.assertRaises(ValueError):
                    MCQ.from_mapping(data)

    def test_rejects_non_string_choices(self):
        data = self.valid_data()
        data["choices"] = ["A", "B", 3, "D"]
        with self.assertRaises(ValueError):
            MCQ.from_mapping(data)


if __name__ == "__main__":
    unittest.main()

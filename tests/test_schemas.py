import unittest

from quizmaker.schemas import MCQ


class MCQSchemaTests(unittest.TestCase):
    def valid_data(self):
        return {
            "question": "What is 2 + 2?",
            "choices": ["1", "2", "4", "5"],
            "answer_index": 2,
            "rationale": "2 + 2 equals 4.",
        }

    def test_accepts_valid_mcq(self):
        mcq = MCQ.from_mapping(self.valid_data())
        self.assertEqual(mcq.question, "What is 2 + 2?")
        self.assertEqual(mcq.answer_index, 2)
        self.assertEqual(mcq.choices, ["1", "2", "4", "5"])

    def test_rejects_invalid_choice_count(self):
        data = self.valid_data()
        data["choices"] = ["A", "B", "C"]
        with self.assertRaises(ValueError):
            MCQ.from_mapping(data)

    def test_rejects_invalid_answer_index(self):
        for answer_index in (-1, 4, "2"):
            with self.subTest(answer_index=answer_index):
                data = self.valid_data()
                data["answer_index"] = answer_index
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

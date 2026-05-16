import unittest

from quizmaker.gemma import extract_json_object


class GemmaParsingTests(unittest.TestCase):
    def test_extracts_plain_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), '{"a": 1}')

    def test_extracts_fenced_json(self):
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_extracts_json_with_extra_text(self):
        self.assertEqual(
            extract_json_object('Here is the JSON:\n{"a": 1}\nDone.'),
            '{"a": 1}',
        )

    def test_rejects_output_without_json_object(self):
        with self.assertRaises(ValueError):
            extract_json_object("not json")


if __name__ == "__main__":
    unittest.main()

"""Small data objects shared by the CLI and future UI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCQ:
    question: str
    choices: list[str]
    answer_index: int
    rationale: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MCQ":
        question = data.get("question")
        choices = data.get("choices")
        answer_index = data.get("answer_index")
        rationale = data.get("rationale")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("choices must be an array of exactly 4 strings")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError("each choice must be a non-empty string")
        if not isinstance(answer_index, int) or not 0 <= answer_index <= 3:
            raise ValueError("answer_index must be an integer from 0 to 3")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")

        return cls(
            question=question.strip(),
            choices=[choice.strip() for choice in choices],
            answer_index=answer_index,
            rationale=rationale.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "choices": self.choices,
            "answer_index": self.answer_index,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ReviewItem:
    id: int
    topic: str
    mcq: MCQ
    priority: int
    cooldown: int
    asked_count: int
    correct_count: int
    wrong_count: int

"""Small data objects shared by the CLI and future UI backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Overview:
    points: list[str]  # each may be "Label: description" or plain text

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Overview":
        points = data.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("points must be a non-empty list")
        if not all(isinstance(p, str) and p.strip() for p in points):
            raise ValueError("each point must be a non-empty string")
        return cls(points=[p.strip() for p in points])

    def to_json(self) -> str:
        return json.dumps({"points": self.points})

    @classmethod
    def from_json(cls, text: str) -> "Overview":
        return cls.from_mapping(json.loads(text))


@dataclass(frozen=True, init=False)
class MCQ:
    question: str
    choices: list[str]
    answer_indices: list[int]
    rationale: str

    def __init__(
        self,
        question: str,
        choices: list[str],
        answer_indices: int | list[int] | None = None,
        rationale: str = "",
        *,
        answer_index: int | None = None,
    ) -> None:
        if answer_indices is None:
            answer_indices = answer_index
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("choices must be an array of exactly 4 strings")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError("each choice must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")

        object.__setattr__(self, "question", question.strip())
        object.__setattr__(self, "choices", [choice.strip() for choice in choices])
        object.__setattr__(
            self,
            "answer_indices",
            self._normalize_answer_indices(answer_indices),
        )
        object.__setattr__(self, "rationale", rationale.strip())

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MCQ":
        question = data.get("question")
        choices = data.get("choices")
        answer_indices = data.get("answer_indices")
        if answer_indices is None:
            answer_indices = data.get("answer_index")
        rationale = data.get("rationale")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("choices must be an array of exactly 4 strings")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError("each choice must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")

        return cls(
            question=question,
            choices=choices,
            answer_indices=answer_indices,
            rationale=rationale,
        )

    @staticmethod
    def _normalize_answer_indices(value: int | list[int]) -> list[int]:
        if isinstance(value, bool):
            raise ValueError("answer_indices must contain integers from 0 to 3")
        if isinstance(value, int):
            indices = [value]
        elif isinstance(value, list):
            indices = value
        else:
            raise ValueError("answer_indices must be a non-empty array of integers from 0 to 3")

        if not indices:
            raise ValueError("answer_indices must not be empty")
        if len(indices) >= 4:
            raise ValueError("answer_indices must leave at least one incorrect choice")
        normalized: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 3:
                raise ValueError("answer_indices must contain integers from 0 to 3")
            if index in normalized:
                raise ValueError("answer_indices must not contain duplicates")
            normalized.append(index)
        return sorted(normalized)

    @property
    def answer_index(self) -> int:
        """Legacy single-answer accessor used by older tests and scripts."""
        return self.answer_indices[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "choices": self.choices,
            "answer_indices": self.answer_indices,
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

"""Straight-line M1 learning loop without LangGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quizmaker.schemas import MCQ, ReviewItem
from quizmaker.storage import QuizStore


class QuizGenerator(Protocol):
    def generate_overview(self, topic: str) -> str:
        ...

    def generate_quiz(self, topic: str, overview: str, count: int = 3) -> list[MCQ]:
        ...


@dataclass
class AskedQuestion:
    item_id: int
    mcq: MCQ
    is_review: bool


@dataclass(frozen=True)
class AnswerResult:
    item_id: int
    is_correct: bool
    correct_index: int
    rationale: str


class CoreLoop:
    def __init__(self, store: QuizStore, generator: QuizGenerator, review_every: int = 3) -> None:
        if review_every < 1:
            raise ValueError("review_every must be at least 1")
        self.store = store
        self.generator = generator
        self.review_every = review_every
        session = store.load_session()
        self.topic = session["topic"]
        self.overview = session["overview"]
        self.turn_count = session["turn_count"]

    def start_topic(self, topic: str, quiz_count: int = 3) -> tuple[str, list[AskedQuestion]]:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("topic cannot be empty")

        self.overview = self.generator.generate_overview(self.topic)
        mcqs = self.generator.generate_quiz(self.topic, self.overview, quiz_count)
        item_ids = self.store.add_quiz_items(self.topic, mcqs)
        self.turn_count = 0
        self.store.save_session(self.topic, self.overview, self.turn_count)
        self.store.add_history("assistant", "overview", self.overview)
        return self.overview, [
            AskedQuestion(item_id=item_id, mcq=mcq, is_review=False)
            for item_id, mcq in zip(item_ids, mcqs)
        ]

    def answer(self, asked: AskedQuestion, choice_index: int) -> tuple[bool, str]:
        result = self.answer_item(asked.item_id, choice_index)
        return result.is_correct, result.rationale

    def answer_item(self, item_id: int, choice_index: int) -> AnswerResult:
        if not 0 <= choice_index <= 3:
            raise ValueError("choice_index must be 0-3")

        item = self.store.get_quiz_item(item_id)
        if item is None:
            raise ValueError(f"quiz item not found: {item_id}")

        is_correct = choice_index == item.mcq.answer_index
        self.store.record_answer(item.id, is_correct)
        if not is_correct:
            self.store.mark_wrong_for_review(item.id)
        self.store.add_history(
            "user",
            "answer",
            f"item={item.id} choice={choice_index} correct={is_correct}",
        )
        return AnswerResult(
            item_id=item.id,
            is_correct=is_correct,
            correct_index=item.mcq.answer_index,
            rationale=item.mcq.rationale,
        )

    def next_turn(self) -> AskedQuestion | None:
        self.turn_count += 1
        self.store.decrement_cooldowns()
        self.store.save_session(self.topic, self.overview, self.turn_count)

        if self.turn_count % self.review_every != 0:
            return None

        review_item = self.store.due_review_item()
        if review_item is None:
            return None
        self.store.add_history("assistant", "review", f"item={review_item.id}")
        return self._to_asked(review_item)

    def _to_asked(self, item: ReviewItem) -> AskedQuestion:
        return AskedQuestion(item_id=item.id, mcq=item.mcq, is_review=True)

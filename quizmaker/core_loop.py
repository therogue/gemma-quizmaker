"""M2 learning loop shell backed by LangGraph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from quizmaker.schemas import MCQ, Overview, ReviewItem
from quizmaker.storage import QuizStore


GRAPH_NODE_NAMES = (
    "overview",
    "quiz_gen",
    "verify",
    "safety",
    "grade",
    "review_inject",
)


class QuizGenerator(Protocol):
    def generate_overview(self, topic: str) -> Overview:
        ...

    def generate_quiz(self, topic: str, overview: str, count: int = 3) -> list[MCQ]:
        ...


class QuizVerifier(Protocol):
    def verify_mcq(self, topic: str, overview: str, mcq: MCQ) -> bool:
        ...


class SafetyChecker(Protocol):
    def is_safe(self, topic: str, overview: str, mcqs: list[MCQ]) -> bool:
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


class CoreLoopState(TypedDict, total=False):
    operation: str
    topic: str
    overview: str
    quiz_count: int
    mcqs: list[MCQ]
    item_id: int
    choice_index: int
    result: AnswerResult
    review: AskedQuestion | None


class AcceptAllVerifier:
    def verify_mcq(self, topic: str, overview: str, mcq: MCQ) -> bool:
        return True


class AllowAllSafetyChecker:
    def is_safe(self, topic: str, overview: str, mcqs: list[MCQ]) -> bool:
        return True


class CoreLoop:
    def __init__(
        self,
        store: QuizStore,
        generator: QuizGenerator,
        review_every: int = 3,
        log_path: Path | None = None,
        verifier: QuizVerifier | None = None,
        safety_checker: SafetyChecker | None = None,
    ) -> None:
        if review_every < 1:
            raise ValueError("review_every must be at least 1")
        self.store = store
        self.generator = generator
        self.review_every = review_every
        self.log_path = log_path
        self.verifier = verifier or AcceptAllVerifier()
        self.safety_checker = safety_checker or AllowAllSafetyChecker()
        self.graph_node_names = GRAPH_NODE_NAMES
        self.graph = self._build_graph()
        session = store.load_session()
        self.topic = session["topic"]
        overview_raw = session["overview"]
        try:
            self.overview: Overview | None = Overview.from_json(overview_raw) if overview_raw else None
        except (ValueError, Exception):
            self.overview = None
        self.turn_count = session["turn_count"]

    def _build_graph(self):
        graph = StateGraph(CoreLoopState)
        graph.add_node("overview", self._overview_node)
        graph.add_node("quiz_gen", self._quiz_gen_node)
        graph.add_node("verify", self._verify_node)
        graph.add_node("safety", self._safety_node)
        graph.add_node("grade", self._grade_node)
        graph.add_node("review_inject", self._review_inject_node)

        graph.add_conditional_edges(
            START,
            self._route_operation,
            {
                "start_topic": "overview",
                "answer": "grade",
                "next_turn": "review_inject",
            },
        )
        graph.add_edge("overview", "quiz_gen")
        graph.add_edge("quiz_gen", "verify")
        graph.add_edge("verify", "safety")
        graph.add_edge("safety", END)
        graph.add_edge("grade", END)
        graph.add_edge("review_inject", END)
        return graph.compile()

    def _route_operation(self, state: CoreLoopState) -> str:
        return state["operation"]

    def _overview_node(self, state: CoreLoopState) -> CoreLoopState:
        output: CoreLoopState = {
            "overview": self.generator.generate_overview(state["topic"]).to_json(),
        }
        self._log_node("overview", state, output)
        return output

    def _quiz_gen_node(self, state: CoreLoopState) -> CoreLoopState:
        output: CoreLoopState = {
            "mcqs": self.generator.generate_quiz(
                state["topic"],
                state["overview"],
                state["quiz_count"],
            )
        }
        self._log_node("quiz_gen", state, output)
        return output

    def _verify_node(self, state: CoreLoopState) -> CoreLoopState:
        verified: list[MCQ] = []
        for mcq in state["mcqs"]:
            if self.verifier.verify_mcq(state["topic"], state["overview"], mcq):
                verified.append(mcq)
                continue

            retry = self.generator.generate_quiz(state["topic"], state["overview"], 1)
            if retry and self.verifier.verify_mcq(state["topic"], state["overview"], retry[0]):
                verified.append(retry[0])

        output: CoreLoopState = {"mcqs": verified}
        self._log_node("verify", state, output)
        return output

    def _safety_node(self, state: CoreLoopState) -> CoreLoopState:
        if self.safety_checker.is_safe(state["topic"], state["overview"], state["mcqs"]):
            output: CoreLoopState = {
                "overview": state["overview"],
                "mcqs": state["mcqs"],
            }
        else:
            output = {
                "overview": Overview(points=["I cannot help with that topic."]).to_json(),
                "mcqs": [],
            }
        self._log_node("safety", state, output)
        return output

    def _grade_node(self, state: CoreLoopState) -> CoreLoopState:
        item_id = state["item_id"]
        choice_index = state["choice_index"]
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
        output: CoreLoopState = {
            "result": AnswerResult(
                item_id=item.id,
                is_correct=is_correct,
                correct_index=item.mcq.answer_index,
                rationale=item.mcq.rationale,
            )
        }
        self._log_node("grade", state, output)
        return output

    def _review_inject_node(self, state: CoreLoopState) -> CoreLoopState:
        self.turn_count += 1
        self.store.decrement_cooldowns()
        self.store.save_session(self.topic, self.overview.to_json() if self.overview else "", self.turn_count)

        review: AskedQuestion | None = None
        if self.turn_count % self.review_every == 0:
            review_item = self.store.due_review_item()
            if review_item is not None:
                self.store.add_history("assistant", "review", f"item={review_item.id}")
                review = self._to_asked(review_item)

        output: CoreLoopState = {"review": review}
        self._log_node("review_inject", state, output)
        return output

    def _log_node(self, node: str, input_state: CoreLoopState, output_state: CoreLoopState) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "node": node,
            "input": self._jsonable(input_state),
            "output": self._jsonable(output_state),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, MCQ):
            return {
                "question": value.question,
                "choices": value.choices,
                "answer_index": value.answer_index,
                "rationale": value.rationale,
            }
        if isinstance(value, AskedQuestion):
            return {
                "item_id": value.item_id,
                "mcq": self._jsonable(value.mcq),
                "is_review": value.is_review,
            }
        if isinstance(value, AnswerResult):
            return {
                "item_id": value.item_id,
                "is_correct": value.is_correct,
                "correct_index": value.correct_index,
                "rationale": value.rationale,
            }
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value

    def start_topic(self, topic: str, quiz_count: int = 3) -> tuple[Overview, list[AskedQuestion]]:
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("topic cannot be empty")

        state = self.graph.invoke(
            {
                "operation": "start_topic",
                "topic": self.topic,
                "quiz_count": quiz_count,
            }
        )
        overview_json = state["overview"]
        self.overview = Overview.from_json(overview_json)
        mcqs = state["mcqs"]
        item_ids = self.store.add_quiz_items(self.topic, mcqs)
        self.turn_count = 0
        self.store.save_session(self.topic, overview_json, self.turn_count)
        self.store.add_history("assistant", "overview", overview_json)
        return self.overview, [
            AskedQuestion(item_id=item_id, mcq=mcq, is_review=False)
            for item_id, mcq in zip(item_ids, mcqs)
        ]

    def answer(self, asked: AskedQuestion, choice_index: int) -> tuple[bool, str]:
        result = self.answer_item(asked.item_id, choice_index)
        return result.is_correct, result.rationale

    def answer_item(self, item_id: int, choice_index: int) -> AnswerResult:
        state = self.graph.invoke(
            {
                "operation": "answer",
                "item_id": item_id,
                "choice_index": choice_index,
            }
        )
        return state["result"]

    def next_turn(self) -> AskedQuestion | None:
        state = self.graph.invoke({"operation": "next_turn"})
        return state["review"]

    def _to_asked(self, item: ReviewItem) -> AskedQuestion:
        return AskedQuestion(item_id=item.id, mcq=item.mcq, is_review=True)

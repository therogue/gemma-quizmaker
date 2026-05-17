"""M2 learning loop shell backed by LangGraph."""

from __future__ import annotations

import json
import re
from inspect import Parameter, signature
from dataclasses import dataclass
from difflib import SequenceMatcher
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

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "about",
    "according",
    "answer",
    "best",
    "choice",
    "correct",
    "does",
    "from",
    "following",
    "into",
    "most",
    "question",
    "the",
    "these",
    "this",
    "what",
    "which",
    "would",
}


class QuizGenerator(Protocol):
    def generate_overview(self, topic: str) -> Overview:
        ...

    def generate_quiz(
        self,
        topic: str,
        overview: str,
        count: int = 3,
        avoid_questions: list[str] | None = None,
    ) -> list[MCQ]:
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
    topic: str


@dataclass(frozen=True)
class AnswerResult:
    item_id: int
    is_correct: bool
    correct_indices: list[int]
    rationale: str
    review: AskedQuestion | None = None

    @property
    def correct_index(self) -> int:
        """Legacy single-answer accessor for older callers."""
        return self.correct_indices[0]


class CoreLoopState(TypedDict, total=False):
    operation: str
    conversation_id: int
    topic: str
    overview: str
    quiz_count: int
    mcqs: list[MCQ]
    item_id: int
    choice_index: int
    choice_indices: list[int]
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
                "more_questions": "quiz_gen",
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
        avoid_questions = self.store.list_quiz_questions(
            state["conversation_id"], state.get("topic")
        )
        output: CoreLoopState = {
            "mcqs": self._generate_quiz(
                state["topic"],
                state["overview"],
                state["quiz_count"],
                avoid_questions=avoid_questions,
            )
        }
        self._log_node("quiz_gen", state, output)
        return output

    def _verify_node(self, state: CoreLoopState) -> CoreLoopState:
        verified: list[MCQ] = []
        existing_questions = self.store.list_quiz_questions(
            state["conversation_id"], state.get("topic")
        )
        for mcq in state["mcqs"]:
            if self._verify_mcq_quality(state, mcq, verified, existing_questions):
                verified.append(mcq)
                continue

            retry = self._generate_quiz(
                state["topic"],
                state["overview"],
                1,
                avoid_questions=existing_questions + [item.question for item in verified],
            )
            if retry and self._verify_mcq_quality(
                state, retry[0], verified, existing_questions
            ):
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
        conversation_id = state["conversation_id"]
        item_id = state["item_id"]
        choice_indices = self._normalize_choice_indices(
            state.get("choice_indices", state.get("choice_index"))
        )

        item = self.store.get_quiz_item(item_id, conversation_id)
        if item is None:
            raise ValueError(f"quiz item {item_id} not found in conversation {conversation_id}")

        is_correct = set(choice_indices) == set(item.mcq.answer_indices)
        self.store.record_answer(item.id, conversation_id, is_correct)
        if not is_correct:
            self.store.mark_wrong_for_review(item.id, conversation_id)
        self.store.add_message(
            conversation_id,
            "user",
            "answer",
            json.dumps(
                {
                    "item_id": item.id,
                    "choice_indices": choice_indices,
                    "choice_index": choice_indices[0],
                    "correct": is_correct,
                    "correct_indices": item.mcq.answer_indices,
                    "correct_index": item.mcq.answer_index,
                    "rationale": item.mcq.rationale,
                }
            ),
        )
        review = self._advance_turn(conversation_id, exclude_item_id=item.id)
        output: CoreLoopState = {
            "result": AnswerResult(
                item_id=item.id,
                is_correct=is_correct,
                correct_indices=item.mcq.answer_indices,
                rationale=item.mcq.rationale,
                review=review,
            )
        }
        self._log_node("grade", state, output)
        return output

    def _review_inject_node(self, state: CoreLoopState) -> CoreLoopState:
        conversation_id = state["conversation_id"]
        review = self._advance_turn(conversation_id)
        output: CoreLoopState = {"review": review}
        self._log_node("review_inject", state, output)
        return output

    def _log_node(self, node: str, input_state: CoreLoopState, output_state: CoreLoopState) -> None:
        record = {
            "node": node,
            "input": self._jsonable(input_state),
            "output": self._jsonable(output_state),
        }
        conversation_id = input_state.get("conversation_id")
        if conversation_id is not None:
            self.store.add_log(
                f"graph.{node}",
                json.dumps(record, sort_keys=True),
                conversation_id=conversation_id,
                level="debug",
            )
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, MCQ):
            return {
                "question": value.question,
                "choices": value.choices,
                "answer_indices": value.answer_indices,
                "answer_index": value.answer_index,
                "rationale": value.rationale,
            }
        if isinstance(value, AskedQuestion):
            return {
                "item_id": value.item_id,
                "mcq": self._jsonable(value.mcq),
                "is_review": value.is_review,
                "topic": value.topic,
            }
        if isinstance(value, AnswerResult):
            return {
                "item_id": value.item_id,
                "is_correct": value.is_correct,
                "correct_indices": value.correct_indices,
                "correct_index": value.correct_index,
                "rationale": value.rationale,
                "review": self._jsonable(value.review),
            }
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value

    def _verify_mcq_quality(
        self,
        state: CoreLoopState,
        mcq: MCQ,
        accepted: list[MCQ],
        existing_questions: list[str],
    ) -> bool:
        if self._is_too_similar_to_any(
            mcq.question,
            [item.question for item in accepted] + existing_questions,
        ):
            return False
        return self.verifier.verify_mcq(state["topic"], state["overview"], mcq)

    def _generate_quiz(
        self,
        topic: str,
        overview: str,
        count: int,
        *,
        avoid_questions: list[str] | None = None,
    ) -> list[MCQ]:
        generate_quiz = self.generator.generate_quiz
        params = signature(generate_quiz).parameters
        accepts_avoid_questions = (
            "avoid_questions" in params
            or any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
        )
        if accepts_avoid_questions:
            return generate_quiz(
                topic,
                overview,
                count,
                avoid_questions=avoid_questions or [],
            )
        return generate_quiz(topic, overview, count)

    def _is_too_similar_to_any(self, question: str, candidates: list[str]) -> bool:
        normalized = self._normalize_question_text(question)
        question_terms = self._question_terms(question)
        for candidate in candidates:
            candidate_normalized = self._normalize_question_text(candidate)
            if normalized == candidate_normalized:
                return True
            if SequenceMatcher(None, normalized, candidate_normalized).ratio() >= 0.88:
                return True

            candidate_terms = self._question_terms(candidate)
            if not question_terms or not candidate_terms:
                continue
            overlap = len(question_terms & candidate_terms) / len(question_terms | candidate_terms)
            if overlap >= 0.72:
                return True
        return False

    def _normalize_question_text(self, text: str) -> str:
        return " ".join(_WORD_RE.findall(text.lower()))

    def _question_terms(self, text: str) -> set[str]:
        return {
            word
            for word in _WORD_RE.findall(text.lower())
            if len(word) > 2 and word not in _STOP_WORDS
        }

    def _normalize_choice_indices(self, value: Any) -> list[int]:
        if isinstance(value, bool):
            raise ValueError("choice_indices must contain integers from 0 to 3")
        if isinstance(value, int):
            indices = [value]
        elif isinstance(value, list):
            indices = value
        else:
            raise ValueError("choice_indices must be a non-empty array of integers from 0 to 3")

        if not indices:
            raise ValueError("choice_indices must not be empty")
        normalized: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 3:
                raise ValueError("choice_indices must contain integers from 0 to 3")
            if index in normalized:
                raise ValueError("choice_indices must not contain duplicates")
            normalized.append(index)
        return sorted(normalized)

    def start_topic(
        self, conversation_id: int, topic: str, quiz_count: int = 3
    ) -> tuple[Overview, list[AskedQuestion]]:
        topic = topic.strip()
        if not topic:
            raise ValueError("topic cannot be empty")

        # Retire old active items when focus changes to a new topic
        self.store.deactivate_active_items(conversation_id)

        state = self.graph.invoke(
            {
                "operation": "start_topic",
                "conversation_id": conversation_id,
                "topic": topic,
                "quiz_count": quiz_count,
            }
        )
        overview_json = state["overview"]
        overview = Overview.from_json(overview_json)
        mcqs = state["mcqs"]

        item_ids = self.store.add_quiz_items(conversation_id, topic, mcqs)

        # Auto-title from first topic if conversation still has the default title
        conv = self.store.get_conversation(conversation_id)
        if conv["title"] == "New conversation":
            self.store.set_conversation_title(conversation_id, topic)

        self.store.save_conversation(
            conversation_id, topic, overview_json, turn_count=conv["turn_count"]
        )
        self.store.add_message(
            conversation_id, "user", "topic", json.dumps({"text": topic}),
        )
        self.store.add_message(
            conversation_id, "assistant", "overview",
            json.dumps({"topic": topic, "points": overview.points}),
        )
        for item_id, mcq in zip(item_ids, mcqs):
            self.store.add_message(
                conversation_id,
                "assistant",
                "question",
                json.dumps(self._question_message_payload(topic, mcq)),
                quiz_item_id=item_id,
            )

        self._advance_turn(conversation_id, inject_review=False)

        return overview, [
            AskedQuestion(item_id=item_id, mcq=mcq, is_review=False, topic=topic)
            for item_id, mcq in zip(item_ids, mcqs)
        ]

    def answer(
        self, conversation_id: int, asked: AskedQuestion, choice_indices: int | list[int]
    ) -> tuple[bool, str]:
        result = self.answer_item(conversation_id, asked.item_id, choice_indices)
        return result.is_correct, result.rationale

    def answer_item(
        self, conversation_id: int, item_id: int, choice_indices: int | list[int]
    ) -> AnswerResult:
        normalized_choice_indices = self._normalize_choice_indices(choice_indices)
        state = self.graph.invoke(
            {
                "operation": "answer",
                "conversation_id": conversation_id,
                "item_id": item_id,
                "choice_indices": normalized_choice_indices,
                "choice_index": normalized_choice_indices[0],
            }
        )
        return state["result"]

    def next_turn(self, conversation_id: int) -> AskedQuestion | None:
        state = self.graph.invoke(
            {
                "operation": "next_turn",
                "conversation_id": conversation_id,
            }
        )
        return state["review"]

    def more_questions(
        self, conversation_id: int, count: int = 3
    ) -> list[AskedQuestion]:
        """Generate additional quiz items for the current focus topic."""
        conv = self.store.get_conversation(conversation_id)
        topic = conv["topic"]
        overview_json = conv["overview_json"]
        if not topic:
            raise ValueError("conversation has no active topic")

        state = self.graph.invoke(
            {
                "operation": "more_questions",
                "conversation_id": conversation_id,
                "topic": topic,
                "overview": overview_json,
                "quiz_count": count,
            }
        )
        mcqs = state["mcqs"]
        item_ids = self.store.add_quiz_items(conversation_id, topic, mcqs)
        for item_id, mcq in zip(item_ids, mcqs):
            self.store.add_message(
                conversation_id,
                "assistant",
                "question",
                json.dumps(self._question_message_payload(topic, mcq)),
                quiz_item_id=item_id,
            )
        return [
            AskedQuestion(item_id=item_id, mcq=mcq, is_review=False, topic=topic)
            for item_id, mcq in zip(item_ids, mcqs)
        ]

    def suggest_topics(
        self, conversation_id: int, use_history: bool = True, count: int = 4
    ) -> list[str]:
        """Return related topic suggestions, optionally using conversation history.

        Returns empty list if the generator doesn't support topic suggestion.
        """
        suggest_fn = getattr(self.generator, "suggest_topics", None)
        if suggest_fn is None:
            return []
        conv = self.store.get_conversation(conversation_id)
        history = (
            self.store.get_messages(conversation_id, limit=20) if use_history else None
        )
        return suggest_fn(conv["topic"], conv["overview_json"], history=history, count=count)

    def chat(
        self, conversation_id: int, user_text: str
    ) -> tuple[str, AskedQuestion | None]:
        """Send a free-form message and get a plain Gemma reply.

        Returns (reply_text, review_question_or_None). A review question is
        included when this chat turn crosses the review_every threshold.
        """
        conv = self.store.get_conversation(conversation_id)

        # Load last 10 messages for context before storing the new one
        history = self.store.get_messages(conversation_id, limit=10)

        self.store.add_message(
            conversation_id, "user", "chat", json.dumps({"text": user_text})
        )

        reply = self.generator.generate_chat_reply(  # type: ignore[attr-defined]
            conv["topic"], conv["overview_json"], history, user_text
        )

        self.store.add_message(
            conversation_id, "assistant", "chat", json.dumps({"text": reply})
        )

        review = self._advance_turn(conversation_id)

        return reply, review

    def _advance_turn(
        self,
        conversation_id: int,
        *,
        exclude_item_id: int | None = None,
        inject_review: bool = True,
    ) -> AskedQuestion | None:
        conv = self.store.get_conversation(conversation_id)
        turn_count = conv["turn_count"] + 1
        self.store.decrement_cooldowns(conversation_id, exclude_item_id=exclude_item_id)
        self.store.save_conversation(
            conversation_id, conv["topic"], conv["overview_json"], turn_count
        )

        if not inject_review or turn_count % self.review_every != 0:
            return None

        review_item = self.store.due_review_item(
            conversation_id, exclude_item_id=exclude_item_id
        )
        if review_item is None:
            return None

        self.store.activate_for_review(review_item.id, conversation_id)
        self.store.add_message(
            conversation_id,
            "assistant",
            "review",
            json.dumps({"item_id": review_item.id, "topic": review_item.topic}),
            quiz_item_id=review_item.id,
        )
        return self._to_asked(review_item)

    def _question_message_payload(self, topic: str, mcq: MCQ) -> dict[str, Any]:
        return {
            "topic": topic,
            "question": mcq.question,
            "choices": mcq.choices,
        }

    def _to_asked(self, item: ReviewItem) -> AskedQuestion:
        return AskedQuestion(item_id=item.id, mcq=item.mcq, is_review=True, topic=item.topic)

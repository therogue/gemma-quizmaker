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
    topic: str


@dataclass(frozen=True)
class AnswerResult:
    item_id: int
    is_correct: bool
    correct_index: int
    rationale: str
    review: AskedQuestion | None = None


class CoreLoopState(TypedDict, total=False):
    operation: str
    conversation_id: int
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
        conversation_id = state["conversation_id"]
        item_id = state["item_id"]
        choice_index = state["choice_index"]
        if not 0 <= choice_index <= 3:
            raise ValueError("choice_index must be 0-3")

        item = self.store.get_quiz_item(item_id, conversation_id)
        if item is None:
            raise ValueError(f"quiz item {item_id} not found in conversation {conversation_id}")

        is_correct = choice_index == item.mcq.answer_index
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
                    "choice_index": choice_index,
                    "correct": is_correct,
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
                correct_index=item.mcq.answer_index,
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
                "correct_index": value.correct_index,
                "rationale": value.rationale,
                "review": self._jsonable(value.review),
            }
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value

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
        self, conversation_id: int, asked: AskedQuestion, choice_index: int
    ) -> tuple[bool, str]:
        result = self.answer_item(conversation_id, asked.item_id, choice_index)
        return result.is_correct, result.rationale

    def answer_item(
        self, conversation_id: int, item_id: int, choice_index: int
    ) -> AnswerResult:
        state = self.graph.invoke(
            {
                "operation": "answer",
                "conversation_id": conversation_id,
                "item_id": item_id,
                "choice_index": choice_index,
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

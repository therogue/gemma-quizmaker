"""Dev stub server for UI development — real storage + fake Gemma responses.

Uses the real QuizStore and CoreLoop with a FakeGenerator so all storage,
routing, and conversation management code runs for real, but no GPU is needed.

Run with:
    uv run uvicorn scripts.stub_server:app --reload --port 8001
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quizmaker.core_loop import (
    AcceptAllVerifier,
    AllowAllSafetyChecker,
    AskedQuestion,
    CoreLoop,
)
from quizmaker.schemas import MCQ, Overview
from quizmaker.storage import QuizStore

# ── fake generator ────────────────────────────────────────────────────────────

class FakeGenerator:
    def generate_overview(self, topic: str) -> Overview:
        return Overview(points=[
            f"Definition: {topic} is the subject of this study session",
            "Concept one: first key idea about this topic",
            "Concept two: second key idea about this topic",
            "Concept three: third key idea about this topic",
        ])

    def generate_quiz(
        self,
        topic: str,
        overview: str,
        count: int = 3,
        avoid_questions: list[str] | None = None,
    ) -> list[MCQ]:
        templates = [
            (
                f"[Stub Q1] Which statements accurately describe the definition of {topic}?",
                [
                    f"{topic} is the focus of this study session",
                    f"{topic} connects to several key concepts",
                    f"{topic} has no practical examples",
                    f"{topic} can only be learned by memorizing dates",
                ],
                [0, 1],
                f"The first two choices are correct because they accurately describe {topic}.",
            ),
            (
                f"[Stub Q2] Which example best applies a key idea from {topic}?",
                [
                    f"Using {topic} to explain a real scenario",
                    f"Ignoring all context around {topic}",
                    f"Treating {topic} as unrelated facts only",
                    f"Replacing {topic} with an unrelated subject",
                ],
                [0],
                f"The first choice is correct because it applies {topic} to a concrete scenario.",
            ),
            (
                f"[Stub Q3] Which statements are common misconceptions about {topic}?",
                [
                    f"{topic} never changes with context",
                    f"{topic} can involve multiple connected ideas",
                    f"{topic} is always solved by one memorized phrase",
                    f"{topic} can be explained with examples",
                ],
                [0, 2],
                f"The first and third choices are misconceptions about {topic}.",
            ),
            (
                f"[Stub Q4] Which comparison statements about {topic} are accurate?",
                [
                    f"{topic} can be compared across different examples",
                    f"{topic} always has one fixed context",
                    f"{topic} may share patterns with related ideas",
                    f"{topic} should never be connected to prior knowledge",
                ],
                [0, 2],
                f"The first and third choices are accurate comparisons for {topic}.",
            ),
            (
                f"[Stub Q5] Which steps help explain {topic} clearly?",
                [
                    "Start with a concrete example",
                    "Define the important terms",
                    "Skip the relationship between ideas",
                    "Use only unrelated trivia",
                ],
                [0, 1],
                f"Concrete examples and clear definitions help explain {topic}.",
            ),
            (
                f"[Stub Q6] Which cause-and-effect statements fit {topic}?",
                [
                    f"Changing context can change how {topic} is applied",
                    f"{topic} has no causes or effects",
                    f"Examples can reveal consequences of {topic}",
                    f"Evidence is irrelevant when studying {topic}",
                ],
                [0, 2],
                f"The first and third choices describe useful cause-and-effect thinking for {topic}.",
            ),
        ]
        avoided = set(avoid_questions or [])
        available = [template for template in templates if template[0] not in avoided]
        if not available:
            return []
        return [
            MCQ(
                question=available[i % len(available)][0],
                choices=available[i % len(available)][1],
                answer_indices=available[i % len(available)][2],
                rationale=available[i % len(available)][3],
            )
            for i in range(min(count, len(available)))
        ]

    def generate_chat_reply(
        self, topic: str, overview_json: str, history: list[dict], user_text: str
    ) -> str:
        return (
            f"[Stub reply] You asked about '{user_text}' in the context of {topic}. "
            "In a real session, Gemma would answer this question based on the overview and "
            "conversation history. This is a stub response for UI development."
        )

    def suggest_topics(
        self,
        topic: str,
        overview_json: str,
        history: list[dict] | None = None,
        count: int = 4,
    ) -> list[str]:
        return [
            f"{topic} — advanced concepts",
            f"{topic} — historical context",
            f"Applications of {topic}",
            f"{topic} and related fields",
        ][:count]


# ── app setup ─────────────────────────────────────────────────────────────────

_DB_PATH = Path("data/stub_quizmaker.sqlite3")
_store = QuizStore(_DB_PATH)
_loop = CoreLoop(
    _store,
    FakeGenerator(),
    verifier=AcceptAllVerifier(),
    safety_checker=AllowAllSafetyChecker(),
    review_every=3,
)

app = FastAPI(title="gemma-quizmaker (stub)", version="0.1.0")

_STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ── request / response models ─────────────────────────────────────────────────

class ConversationOut(BaseModel):
    id: int
    title: str
    topic: str
    turn_count: int
    created_at: str
    updated_at: str


class OverviewOut(BaseModel):
    points: list[str]


class QuestionOut(BaseModel):
    item_id: int
    is_review: bool
    topic: str
    question: str
    choices: list[str]


class ConversationDetailOut(BaseModel):
    id: int
    title: str
    topic: str
    turn_count: int
    created_at: str
    updated_at: str
    overview: OverviewOut | None
    messages: list[dict]
    active_questions: list[QuestionOut]


class StartTopicRequest(BaseModel):
    topic: str
    quiz_count: int = 3


class StartTopicResponse(BaseModel):
    overview: OverviewOut
    questions: list[QuestionOut]


class AnswerRequest(BaseModel):
    item_id: int
    choice_index: int | None = None
    choice_indices: list[int] | None = None


class AnswerResponse(BaseModel):
    item_id: int
    is_correct: bool
    correct_index: int
    correct_indices: list[int]
    rationale: str
    review: QuestionOut | None = None


class TurnResponse(BaseModel):
    review: QuestionOut | None


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    review: QuestionOut | None


class MessageRequest(BaseModel):
    text: str


class MessageResponse(BaseModel):
    overview: OverviewOut | None = None
    questions: list[QuestionOut] | None = None
    reply: str | None = None
    review: QuestionOut | None = None


class ActionRequest(BaseModel):
    action: str
    count: int = 3


class MoreQuestionsResponse(BaseModel):
    questions: list[QuestionOut]


class SuggestTopicsResponse(BaseModel):
    suggestions: list[str]


# Fix forward references
ConversationDetailOut.model_rebuild()

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_conversation_or_404(conversation_id: int) -> dict:
    conv = _store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    return conv


def _question_out(asked: AskedQuestion) -> QuestionOut:
    return QuestionOut(
        item_id=asked.item_id,
        is_review=asked.is_review,
        topic=asked.topic,
        question=asked.mcq.question,
        choices=asked.mcq.choices,
    )


def _overview_from_json(overview_json: str) -> OverviewOut | None:
    if not overview_json:
        return None
    try:
        return OverviewOut(points=Overview.from_json(overview_json).points)
    except Exception:
        return None


# ── conversation endpoints ────────────────────────────────────────────────────

@app.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation() -> ConversationOut:
    conv_id = _store.create_conversation()
    conv = _store.get_conversation(conv_id)
    return ConversationOut(**{k: conv[k] for k in ConversationOut.model_fields})


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations() -> list[ConversationOut]:
    convs = _store.list_conversations()
    return [ConversationOut(**{k: c[k] for k in ConversationOut.model_fields}) for c in convs]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(conversation_id: int) -> ConversationDetailOut:
    conv = _get_conversation_or_404(conversation_id)
    messages = _store.get_messages(conversation_id)
    active_items = _store.get_active_quiz_items(conversation_id)
    return ConversationDetailOut(
        id=conv["id"],
        title=conv["title"],
        topic=conv["topic"],
        turn_count=conv["turn_count"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        overview=_overview_from_json(conv["overview_json"]),
        messages=[dict(m) for m in messages],
        active_questions=[
            QuestionOut(
                item_id=item.id,
                is_review=item.priority > 0,
                topic=item.topic,
                question=item.mcq.question,
                choices=item.mcq.choices,
            )
            for item in active_items
        ],
    )


# ── conversation-scoped study endpoints ───────────────────────────────────────

@app.post("/conversations/{conversation_id}/start-topic", response_model=StartTopicResponse)
def start_topic(conversation_id: int, body: StartTopicRequest) -> StartTopicResponse:
    _get_conversation_or_404(conversation_id)
    try:
        overview, questions = _loop.start_topic(conversation_id, body.topic, quiz_count=body.quiz_count)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StartTopicResponse(
        overview=OverviewOut(points=overview.points),
        questions=[_question_out(q) for q in questions],
    )


@app.post("/conversations/{conversation_id}/answer", response_model=AnswerResponse)
def answer(conversation_id: int, body: AnswerRequest) -> AnswerResponse:
    _get_conversation_or_404(conversation_id)
    choice_indices = body.choice_indices
    if choice_indices is None:
        choice_indices = [body.choice_index] if body.choice_index is not None else []
    try:
        result = _loop.answer_item(conversation_id, body.item_id, choice_indices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return AnswerResponse(
        item_id=result.item_id,
        is_correct=result.is_correct,
        correct_index=result.correct_index,
        correct_indices=result.correct_indices,
        rationale=result.rationale,
        review=_question_out(result.review) if result.review else None,
    )


@app.post("/conversations/{conversation_id}/turn", response_model=TurnResponse)
def turn(conversation_id: int) -> TurnResponse:
    _get_conversation_or_404(conversation_id)
    review = _loop.next_turn(conversation_id)
    return TurnResponse(review=_question_out(review) if review else None)


@app.post("/conversations/{conversation_id}/chat", response_model=ChatResponse)
def chat(conversation_id: int, body: ChatRequest) -> ChatResponse:
    _get_conversation_or_404(conversation_id)
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    reply, review = _loop.chat(conversation_id, body.text.strip())
    return ChatResponse(
        reply=reply,
        review=_question_out(review) if review else None,
    )


_CHAT_STARTERS = {
    "what", "why", "how", "when", "where", "who", "explain", "tell",
    "can", "is", "are", "does", "do", "did", "describe", "compare", "give",
    "could", "would", "should", "define", "list", "name",
}


def _is_chat_intent(text: str) -> bool:
    if text.endswith("?"):
        return True
    first = text.split()[0].lower().rstrip("?!,.")
    return first in _CHAT_STARTERS


@app.post("/conversations/{conversation_id}/message", response_model=MessageResponse)
def message(conversation_id: int, body: MessageRequest) -> MessageResponse:
    _get_conversation_or_404(conversation_id)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text cannot be empty")
    if _is_chat_intent(text):
        reply, review = _loop.chat(conversation_id, text)
        return MessageResponse(reply=reply, review=_question_out(review) if review else None)
    try:
        overview, questions = _loop.start_topic(conversation_id, text, quiz_count=3)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MessageResponse(
        overview=OverviewOut(points=overview.points),
        questions=[_question_out(q) for q in questions],
    )


@app.post("/conversations/{conversation_id}/actions")
def actions(conversation_id: int, body: ActionRequest):
    _get_conversation_or_404(conversation_id)

    if body.action == "more_questions":
        try:
            questions = _loop.more_questions(conversation_id, count=body.count)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return MoreQuestionsResponse(questions=[_question_out(q) for q in questions])

    if body.action == "suggest_topics":
        suggestions = _loop.suggest_topics(conversation_id, use_history=True, count=body.count)
        return SuggestTopicsResponse(suggestions=suggestions)

    raise HTTPException(status_code=422, detail=f"unknown action: {body.action!r}")

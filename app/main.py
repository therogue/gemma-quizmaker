from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quizmaker.core_loop import AskedQuestion, CoreLoop
from quizmaker.gemma import GemmaQuizGenerator, load_model
from quizmaker.schemas import Overview
from quizmaker.storage import QuizStore

_DB_PATH = Path(os.environ.get("QUIZMAKER_DB", "data/quizmaker.sqlite3"))
_REVIEW_EVERY = int(os.environ.get("QUIZMAKER_REVIEW_EVERY", "3"))

_store: QuizStore
_loop: CoreLoop


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _loop
    model, processor = load_model()
    _store = QuizStore(_DB_PATH)
    _loop = CoreLoop(_store, GemmaQuizGenerator(model, processor), review_every=_REVIEW_EVERY)
    yield
    _store.close()


app = FastAPI(title="gemma-quizmaker", version="0.1.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ── request / response models ────────────────────────────────────────────────

class ConversationOut(BaseModel):
    id: int
    title: str
    topic: str
    turn_count: int
    created_at: str
    updated_at: str


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


class OverviewOut(BaseModel):
    points: list[str]


class QuestionOut(BaseModel):
    item_id: int
    is_review: bool
    question: str
    choices: list[str]


class StartTopicRequest(BaseModel):
    topic: str
    quiz_count: int = 3


class StartTopicResponse(BaseModel):
    overview: OverviewOut
    questions: list[QuestionOut]


class AnswerRequest(BaseModel):
    item_id: int
    choice_index: int


class AnswerResponse(BaseModel):
    item_id: int
    is_correct: bool
    correct_index: int
    rationale: str


class TurnResponse(BaseModel):
    review: QuestionOut | None


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    review: QuestionOut | None


# Fix forward references after all models are defined
ConversationDetailOut.model_rebuild()


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_conversation_or_404(conversation_id: int) -> dict:
    conv = _store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    return conv


def _question_out(asked: AskedQuestion) -> QuestionOut:
    return QuestionOut(
        item_id=asked.item_id,
        is_review=asked.is_review,
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
async def create_conversation() -> ConversationOut:
    conv_id = _store.create_conversation()
    conv = _store.get_conversation(conv_id)
    return ConversationOut(**{k: conv[k] for k in ConversationOut.model_fields})


@app.get("/conversations", response_model=list[ConversationOut])
async def list_conversations() -> list[ConversationOut]:
    convs = _store.list_conversations()
    return [ConversationOut(**{k: c[k] for k in ConversationOut.model_fields}) for c in convs]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(conversation_id: int) -> ConversationDetailOut:
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
                is_review=False,
                question=item.mcq.question,
                choices=item.mcq.choices,
            )
            for item in active_items
        ],
    )


# ── conversation-scoped study endpoints ──────────────────────────────────────

@app.post("/conversations/{conversation_id}/start-topic", response_model=StartTopicResponse)
async def start_topic(conversation_id: int, body: StartTopicRequest) -> StartTopicResponse:
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
async def answer(conversation_id: int, body: AnswerRequest) -> AnswerResponse:
    _get_conversation_or_404(conversation_id)
    try:
        result = _loop.answer_item(conversation_id, body.item_id, body.choice_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return AnswerResponse(
        item_id=result.item_id,
        is_correct=result.is_correct,
        correct_index=result.correct_index,
        rationale=result.rationale,
    )


@app.post("/conversations/{conversation_id}/turn", response_model=TurnResponse)
async def turn(conversation_id: int) -> TurnResponse:
    _get_conversation_or_404(conversation_id)
    review = _loop.next_turn(conversation_id)
    return TurnResponse(review=_question_out(review) if review else None)


@app.post("/conversations/{conversation_id}/chat", response_model=ChatResponse)
async def chat(conversation_id: int, body: ChatRequest) -> ChatResponse:
    _get_conversation_or_404(conversation_id)
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    reply, review = _loop.chat(conversation_id, body.text.strip())
    return ChatResponse(
        reply=reply,
        review=_question_out(review) if review else None,
    )

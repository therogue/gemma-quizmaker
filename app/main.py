from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quizmaker.core_loop import AskedQuestion, CoreLoop
from quizmaker.gemma import GemmaQuizGenerator, load_model
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

class StartTopicRequest(BaseModel):
    topic: str
    quiz_count: int = 3


class QuestionOut(BaseModel):
    item_id: int
    is_review: bool
    question: str
    choices: list[str]


class OverviewOut(BaseModel):
    points: list[str]


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


# ── helpers ──────────────────────────────────────────────────────────────────

def _question_out(asked: AskedQuestion) -> QuestionOut:
    return QuestionOut(
        item_id=asked.item_id,
        is_review=asked.is_review,
        question=asked.mcq.question,
        choices=asked.mcq.choices,
    )


# ── endpoints ────────────────────────────────────────────────────────────────

@app.post("/start-topic", response_model=StartTopicResponse)
async def start_topic(body: StartTopicRequest) -> StartTopicResponse:
    try:
        overview, questions = _loop.start_topic(body.topic, quiz_count=body.quiz_count)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StartTopicResponse(
        overview=OverviewOut(points=overview.points),
        questions=[_question_out(q) for q in questions],
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer(body: AnswerRequest) -> AnswerResponse:
    try:
        result = _loop.answer_item(body.item_id, body.choice_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return AnswerResponse(
        item_id=result.item_id,
        is_correct=result.is_correct,
        correct_index=result.correct_index,
        rationale=result.rationale,
    )


@app.post("/turn", response_model=TurnResponse)
async def turn() -> TurnResponse:
    review = _loop.next_turn()
    return TurnResponse(review=_question_out(review) if review else None)

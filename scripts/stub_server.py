"""Hardcoded stub server for UI development. Delete when task 7 (real FastAPI wire-up) lands."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# --- stub state -----------------------------------------------------------

_item_counter = 0
_items: dict[int, dict] = {}


def _next_id() -> int:
    global _item_counter
    _item_counter += 1
    return _item_counter


# --- request/response models ----------------------------------------------


class StartRequest(BaseModel):
    topic: str
    quiz_count: int = 3


class AnswerRequest(BaseModel):
    item_id: int
    choice_index: int


# --- endpoints ------------------------------------------------------------


@app.post("/start-topic")
def start(req: StartRequest):
    global _item_counter, _items, _turn
    _item_counter = 0
    _items = {}
    _turn = 0

    questions = []
    for i in range(req.quiz_count):
        item_id = _next_id()
        _items[item_id] = {"answer_index": i % 4, "rationale": f"Stub rationale for question {i + 1}."}
        questions.append({
            "item_id": item_id,
            "is_review": False,
            "question": f"[Stub Q{i + 1}] What is a key fact about {req.topic}?",
            "choices": [
                f"Choice A for Q{i + 1}",
                f"Choice B for Q{i + 1}",
                f"Choice C for Q{i + 1}",
                f"Choice D for Q{i + 1}",
            ],
        })

    return {
        "overview": (
            f"Stub overview for '{req.topic}':\n"
            "• Fact one about this topic\n"
            "• Fact two about this topic\n"
            "• Fact three about this topic"
        ),
        "questions": questions,
    }


@app.post("/answer")
def answer(req: AnswerRequest):
    item = _items.get(req.item_id)
    if item is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"quiz item not found: {req.item_id}")
    is_correct = req.choice_index == item["answer_index"]
    return {
        "item_id": req.item_id,
        "is_correct": is_correct,
        "correct_index": item["answer_index"],
        "rationale": item["rationale"],
    }


_turn = 0


@app.post("/turn")
def turn():
    global _turn
    _turn += 1
    if _turn % 3 == 0 and _items:
        item_id = next(iter(_items))
        item = _items[item_id]
        return {
            "review": {
                "item_id": item_id,
                "is_review": True,
                "question": f"[Review] Stub review question for item {item_id}",
                "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
            }
        }
    return {"review": None}

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="gemma-quizmaker", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    return ChatResponse(reply="hello")

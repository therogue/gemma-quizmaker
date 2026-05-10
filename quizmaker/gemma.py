"""Gemma 4 adapter for overview and MCQ generation."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from quizmaker.schemas import MCQ

MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 768


def extract_json_object(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON object found in model output")
    return text[start:end]


def load_model() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if device == "cuda"
        else None
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map=device
    )
    return model, processor


def run_inference(model: Any, processor: Any, messages: list[dict], max_new_tokens: int) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


class GemmaQuizGenerator:
    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def generate_overview(self, topic: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You explain learning topics clearly and concisely.",
            },
            {
                "role": "user",
                "content": (
                    f"Give a short structured overview of this topic: {topic}\n\n"
                    "Use 4-6 compact bullet points. Focus on facts that can be quizzed."
                ),
            },
        ]
        return run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS).strip()

    def generate_quiz(self, topic: str, overview: str, count: int = 3) -> list[MCQ]:
        mcqs: list[MCQ] = []
        for index in range(count):
            mcqs.append(self._generate_one_mcq(topic, overview, index + 1, mcqs))
        return mcqs

    def _generate_one_mcq(
        self, topic: str, overview: str, question_number: int, existing: list[MCQ]
    ) -> MCQ:
        prior_questions = "\n".join(f"- {mcq.question}" for mcq in existing) or "- none"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quiz generator. Output ONLY a valid JSON object with no markdown, "
                    "no comments, and no surrounding text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Overview:\n{overview}\n\n"
                    f"Generate multiple-choice question {question_number}. Avoid these questions:\n"
                    f"{prior_questions}\n\n"
                    "Return a JSON object with exactly these fields:\n"
                    '  "question": string\n'
                    '  "choices": array of exactly 4 strings\n'
                    '  "answer_index": integer 0-3 (index of the correct choice)\n'
                    '  "rationale": string explaining why the answer is correct'
                ),
            },
        ]
        raw = run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS)
        try:
            return MCQ.from_mapping(json.loads(extract_json_object(raw)))
        except Exception as first_error:
            print(f"[retry] MCQ {question_number} failed validation: {first_error}", file=sys.stderr)
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your output was invalid ({first_error}). "
                        "Return ONLY the valid JSON object."
                    ),
                },
            ]
            raw2 = run_inference(self.model, self.processor, retry_messages, MAX_NEW_TOKENS)
            return MCQ.from_mapping(json.loads(extract_json_object(raw2)))

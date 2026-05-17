"""Gemma 4 adapter for overview and MCQ generation."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from quizmaker.schemas import MCQ, Overview

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

    def generate_overview(self, topic: str) -> Overview:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Output ONLY a valid JSON object "
                    "with no markdown, no comments, and no surrounding text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Return a JSON object with exactly this field:\n"
                    '  "points": array of 4-6 strings, each a concrete quiz-worthy fact.\n'
                    '    Prefer the format "Label: description" for each point '
                    '(e.g. "Nucleus: contains protons and neutrons") '
                    "but plain text is acceptable if a label does not fit.\n"
                    "No markdown, no asterisks, plain text only inside the strings."
                ),
            },
        ]
        raw = run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS)
        print(f"[overview] raw output: {raw!r}", file=sys.stderr)
        try:
            return Overview.from_mapping(json.loads(extract_json_object(raw)))
        except Exception as err:
            print(f"[overview retry] failed: {err}", file=sys.stderr)
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your output was invalid ({err}). "
                        "Return ONLY the valid JSON object with no extra text."
                    ),
                },
            ]
            raw2 = run_inference(self.model, self.processor, retry_messages, MAX_NEW_TOKENS)
            print(f"[overview retry] raw output: {raw2!r}", file=sys.stderr)
            return Overview.from_mapping(json.loads(extract_json_object(raw2)))

    def generate_quiz(
        self,
        topic: str,
        overview: str,
        count: int = 3,
        avoid_questions: list[str] | None = None,
    ) -> list[MCQ]:
        mcqs: list[MCQ] = []
        prior_questions = list(avoid_questions or [])
        for index in range(count):
            mcq = self._generate_one_mcq(topic, overview, index + 1, prior_questions)
            mcqs.append(mcq)
            prior_questions.append(mcq.question)
        return mcqs

    def _generate_one_mcq(
        self, topic: str, overview: str, question_number: int, prior_questions: list[str]
    ) -> MCQ:
        prior_question_text = "\n".join(f"- {question}" for question in prior_questions) or "- none"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quiz generator. Output ONLY a valid JSON object with no markdown, "
                    "no comments, and no surrounding text. Questions may have one or more "
                    "correct choices."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Overview:\n{overview}\n\n"
                    f"Generate multiple-choice question {question_number}. Avoid these questions:\n"
                    f"{prior_question_text}\n\n"
                    "Make this question test a distinct overview point, relationship, example, "
                    "or misconception from the prior questions. Do not reuse the same fact "
                    "with different wording.\n\n"
                    "The question may be single-answer or select-all-that-apply. Use multiple "
                    "correct answers when several choices are true, but leave at least one "
                    "choice incorrect. Keep every incorrect choice clearly false.\n\n"
                    "Return a JSON object with exactly these fields:\n"
                    '  "question": string\n'
                    '  "choices": array of exactly 4 strings — do NOT include A/B/C/D labels, the UI adds those\n'
                    '  "answer_indices": non-empty array of integers 0-3 (indices of all correct choices)\n'
                    '  "rationale": string explaining why the answer is correct'
                ),
            },
        ]
        raw = run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS)
        print(f"[mcq {question_number}] raw output: {raw!r}", file=sys.stderr)
        last_error: Exception
        current_messages = messages
        current_raw = raw
        for attempt in range(3):
            try:
                return MCQ.from_mapping(json.loads(extract_json_object(current_raw)))
            except Exception as err:
                last_error = err
                print(
                    f"[retry {attempt + 1}/3] MCQ {question_number} failed: {err}",
                    file=sys.stderr,
                )
                current_messages = current_messages + [
                    {"role": "assistant", "content": current_raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your output was invalid ({err}). "
                            "Return ONLY the valid JSON object with no extra text."
                        ),
                    },
                ]
                current_raw = run_inference(
                    self.model, self.processor, current_messages, MAX_NEW_TOKENS
                )
                print(f"[retry {attempt + 1}/3] raw output: {current_raw!r}", file=sys.stderr)
        raise ValueError(f"MCQ {question_number} failed after 3 attempts: {last_error}")

    def generate_chat_reply(
        self,
        topic: str,
        overview_json: str,
        history: list[dict],
        user_text: str,
    ) -> str:
        """Plain conversational reply. No quiz generation."""
        overview_text = ""
        if overview_json:
            try:
                overview_text = "\n".join(
                    f"- {p}" for p in Overview.from_json(overview_json).points
                )
            except Exception:
                pass

        system_content = f"You are a study assistant helping the user learn about: {topic}."
        if overview_text:
            system_content += f"\n\nCurrent overview:\n{overview_text}"
        system_content += (
            "\n\nAnswer the user's question conversationally, concisely, and educationally. "
            "Do not generate quiz questions unprompted."
        )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for msg in history:
            try:
                text = json.loads(msg["content_json"]).get("text", "")
            except Exception:
                continue
            if msg["kind"] == "chat" and text:
                messages.append({"role": msg["role"], "content": text})

        messages.append({"role": "user", "content": user_text})

        reply = run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS)
        print(f"[chat] raw output: {reply!r}", file=sys.stderr)
        return reply.strip()

    def suggest_topics(
        self,
        topic: str,
        overview_json: str,
        history: list[dict] | None = None,
        count: int = 4,
    ) -> list[str]:
        """Return related topic suggestions. history is optional conversation context."""
        overview_text = ""
        if overview_json:
            try:
                overview_text = "\n".join(
                    f"- {p}" for p in Overview.from_json(overview_json).points
                )
            except Exception:
                pass

        context = f"Topic: {topic}"
        if overview_text:
            context += f"\n\nOverview:\n{overview_text}"

        if history:
            chat_lines = []
            for msg in history:
                try:
                    text = json.loads(msg["content_json"]).get("text", "")
                except Exception:
                    continue
                if msg["kind"] == "chat" and text:
                    prefix = "User" if msg["role"] == "user" else "Assistant"
                    chat_lines.append(f"{prefix}: {text}")
            if chat_lines:
                context += "\n\nRecent conversation:\n" + "\n".join(chat_lines[-6:])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Output ONLY a valid JSON object "
                    "with no markdown, no comments, and no surrounding text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"Suggest {count} related topics the student could explore next. "
                    "Prefer specific sub-topics or closely related concepts over broad subjects. "
                    "If there is conversation history, bias suggestions toward areas the student "
                    "struggled with or asked about.\n\n"
                    'Return a JSON object with exactly this field:\n'
                    '  "suggestions": array of strings, each a short topic name (3-8 words max)'
                ),
            },
        ]
        raw = run_inference(self.model, self.processor, messages, MAX_NEW_TOKENS)
        print(f"[suggest_topics] raw output: {raw!r}", file=sys.stderr)
        try:
            data = json.loads(extract_json_object(raw))
            suggestions = data.get("suggestions", [])
            return [str(s).strip() for s in suggestions if str(s).strip()][:count]
        except Exception as err:
            print(f"[suggest_topics] parse failed: {err}", file=sys.stderr)
            return []

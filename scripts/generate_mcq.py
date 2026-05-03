# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "transformers",
#     "torch",
#     "torchvision",
#     "pillow",
#     "accelerate",
#     "bitsandbytes",
#     "pydantic",
#     "psutil",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
# ///
"""Generate a single validated MCQ for a given topic using Gemma 4 E2B.

Usage:
    uv run scripts/generate_mcq.py "photosynthesis"
"""

import json
import re
import sys

import torch
from pydantic import BaseModel, field_validator
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 512


class MCQ(BaseModel):
    question: str
    choices: list[str]
    answer_index: int
    rationale: str

    @field_validator("choices")
    @classmethod
    def must_have_four_choices(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError(f"expected 4 choices, got {len(v)}")
        return v

    @field_validator("answer_index")
    @classmethod
    def must_be_valid_index(cls, v: int) -> int:
        if not 0 <= v <= 3:
            raise ValueError(f"answer_index must be 0–3, got {v}")
        return v


def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON object found in model output")
    return text[start:end]


def _build_messages(topic: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a quiz generator. "
                "Output ONLY a valid JSON object — no markdown, no explanation, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate one multiple-choice question about: {topic}\n\n"
                "Return a JSON object with exactly these fields:\n"
                '  "question": string\n'
                '  "choices": array of exactly 4 strings\n'
                '  "answer_index": integer 0–3 (index of the correct choice)\n'
                '  "rationale": string explaining why the answer is correct'
            ),
        },
    ]


def _run_inference(model, processor, messages: list[dict]) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def _parse(raw: str) -> MCQ:
    return MCQ(**json.loads(_extract_json(raw)))


def generate_mcq(model, processor, topic: str) -> MCQ:
    messages = _build_messages(topic)
    raw = _run_inference(model, processor, messages)

    try:
        return _parse(raw)
    except Exception as first_error:
        print(f"[retry] first attempt failed: {first_error}", file=sys.stderr)
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Your output was invalid ({first_error}). "
                    "Return ONLY the JSON object, nothing else."
                ),
            },
        ]
        raw2 = _run_inference(model, processor, retry_messages)
        return _parse(raw2)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    ) if device == "cuda" else None
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map=device
    )
    return model, processor


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/generate_mcq.py <topic>", file=sys.stderr)
        sys.exit(1)

    topic = sys.argv[1]
    model, processor = load_model()
    mcq = generate_mcq(model, processor, topic)
    print(json.dumps(mcq.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

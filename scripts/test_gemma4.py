# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers",
#     "torch",
#     "torchvision",
#     "pillow",
#     "accelerate",
#     "psutil",
# ]
# ///
"""Smoke test for Gemma 4 multimodal inference. Mirrors docs/reference/test-google-gemma-4.ipynb."""

import os

import psutil
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E4B-it"
MAX_NEW_TOKENS = 256
ENABLE_THINKING = False


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def report_memory(tag: str, model: torch.nn.Module | None = None) -> None:
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    print(f"[mem:{tag}] process RSS = {_gb(rss)}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            peak = torch.cuda.max_memory_allocated(i)
            name = torch.cuda.get_device_name(i)
            print(
                f"[mem:{tag}] cuda{i} ({name}): allocated={_gb(alloc)} "
                f"reserved={_gb(reserved)} peak={_gb(peak)}"
            )
    if model is not None:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[mem:{tag}] model params={n_params/1e9:.2f}B "
            f"weights={_gb(param_bytes + buffer_bytes)} "
            f"dtype={next(model.parameters()).dtype} "
            f"device={next(model.parameters()).device}"
        )


def main() -> None:
    report_memory("before-load")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    report_memory("after-load", model)

    # 1) Image+text
    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        image_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print("[image+text]")
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    report_memory("after-image-gen", model)

    # 2) Text-only chat
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]
    text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print("[text]")
    print(processor.parse_response(response))
    report_memory("after-text-gen", model)


if __name__ == "__main__":
    main()

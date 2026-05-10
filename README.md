# gemma-quizmaker

Offline, single-user learning agent built on Gemma 4. Explains topics, generates MCQ quizzes, and uses spaced repetition to reinforce retention.

See [`docs/CONCEPT.md`](docs/CONCEPT.md) for the full concept and [`docs/ROADMAP.md`](docs/ROADMAP.md) for the milestone plan.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — handles Python version and all dependencies automatically
- NVIDIA GPU with CUDA 12.x driver (tested on RTX 3070 Laptop, 8 GB VRAM)

> **Note:** Python and torch are pinned to 3.12 / CUDA 12.1 because the default PyTorch wheels target CUDA 13, which requires a newer driver. If your driver supports CUDA 13+, you can relax these pins.

---

## M0 — Smoke test

Verifies that Gemma 4 loads and runs inference on your hardware.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run scripts/test_gemma4.py
```

**Example output:**

```
[mem:before-load] process RSS = 0.48 GB
[mem:before-load] cuda0 (NVIDIA GeForce RTX 3070 Laptop GPU): allocated=0.00 GB reserved=0.00 GB peak=0.00 GB
[mem:after-load] process RSS = 0.90 GB
[mem:after-load] cuda0 (NVIDIA GeForce RTX 3070 Laptop GPU): allocated=6.38 GB reserved=6.41 GB peak=6.38 GB
[mem:after-load] model params=3.94B weights=6.24 GB dtype=torch.bfloat16 device=cuda:0
[image+text]
Based on the image provided, which is a grey, blurred, and largely featureless background,
it is impossible to identify any candy or any animal on it.
[text]
{'role': 'assistant', 'content': 'Here are a few short jokes about saving RAM...'}
```

---

## M0 — MCQ generator

Generates a single validated multiple-choice question for a given topic.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run scripts/generate_mcq.py "photosynthesis"
```

**Example output:**

```json
{
  "question": "What is the primary pigment responsible for capturing light energy for photosynthesis in plants?",
  "choices": [
    "Carotenoids",
    "Anthocyanins",
    "Chlorophyll a",
    "Carotenoids"
  ],
  "answer_index": 2,
  "rationale": "Chlorophyll a is the primary pigment used by plants for capturing light energy during the light-dependent reactions of photosynthesis. While other pigments are involved, chlorophyll a is the main energy absorber."
}
```

The script validates the output with pydantic and retries once on parse or validation failure.

---

## M1 — Backend core loop

Runs the straight-line PoC loop from the roadmap:

- generate a short structured overview for a topic
- generate an N-question MCQ quiz from that overview
- grade answers by exact choice index
- put wrong answers into a SQLite-backed review queue
- interleave due review questions every K turns

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run scripts/run_core_loop.py "photosynthesis"
```

Useful options:

```bash
uv run scripts/run_core_loop.py "photosynthesis" --count 5 --review-every 2 --db data/quizmaker.sqlite3
```

Core backend code lives in [`quizmaker/`](quizmaker/) so the future FastAPI/UI layer can call the same loop instead of reimplementing quiz generation, grading, review scheduling, or persistence.

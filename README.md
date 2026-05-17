# gemma-quizmaker

Offline, single-user learning agent built on Gemma 4. Explains topics, generates MCQ quizzes, and uses spaced repetition to reinforce retention.

See [`docs/CONCEPT.md`](docs/CONCEPT.md) for the full concept,
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the milestone plan, and
[`docs/M3_API_CONTRACT.md`](docs/M3_API_CONTRACT.md) for the current API contract.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — handles Python version and all dependencies automatically
- NVIDIA GPU with CUDA 12.x driver (tested on RTX 3070 Laptop, 8 GB VRAM)

> **Note:** Python and torch are pinned to 3.12 / CUDA 12.1 because the default PyTorch wheels target CUDA 13, which requires a newer driver. If your driver supports CUDA 13+, you can relax these pins.

---

## Setup

```bash
uv sync   # creates .venv and installs all dependencies (once)
```

After that, run any command in either of these equivalent ways:

```bash
uv run <cmd>                 # no activation needed — uv uses .venv automatically
source .venv/bin/activate    # activate once, then run commands directly
```

### CUDA memory (optional)

If you hit CUDA OOM errors, prefix model-loading commands with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True <cmd>
```

---

## Stub server (no GPU required)

For UI development and testing without a GPU, use the stub server. It runs the real storage and routing code but replaces Gemma with a fast fake generator:

```bash
rm -f data/stub_quizmaker.sqlite3   # clear state if needed
uv run uvicorn scripts.stub_server:app --reload --port 8001
```

Once running:

- UI: `http://localhost:8001`
- API docs: `http://localhost:8001/docs`

Hard-refresh the browser (`Ctrl+Shift+R`) if the UI looks like an older version.

---

## UI + API server (GPU required)

```bash
uv run uvicorn app.main:app --port 8000
```

Model loading takes 30–60 seconds on first start. Once ready:

- UI: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`

---

## CLI runner

Runs the loop in a terminal without the UI — useful for fast iteration:

```bash
uv run scripts/run_core_loop.py "photosynthesis"
uv run scripts/run_core_loop.py "photosynthesis" --count 5 --review-every 2 --db data/quizmaker.sqlite3
```

---

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

---

## Debug scripts

Low-level M0 spike scripts, useful for verifying Gemma loads and generates on your hardware:

- `scripts/test_gemma4.py` — smoke test: loads model, runs inference
- `scripts/generate_mcq.py "topic"` — generates and validates a single MCQ

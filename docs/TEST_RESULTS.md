# Test Results

Date: 2026-05-10

Branch: `m1-backend-core-loop`

Commit tested: current `m1-backend-core-loop` branch after the M1 backend test expansion.

## Environment

- Machine: Apple Silicon Mac (`arm64-apple-darwin`)
- System Python: `Python 3.10.0`
- `uv`: `0.11.12`
- Runtime Python used by `uv`: CPython 3.12
- Note: this machine does not have NVIDIA/CUDA available, so the M1 CLI dependency block was kept cross-platform. The older M0 smoke-test scripts remain CUDA-pinned.

## Automated Tests

Command:

```bash
python3 -m unittest discover -s tests
```

Result:

```text
.................
----------------------------------------------------------------------
Ran 17 tests in 0.037s

OK
```

Coverage from these tests:

- MCQ schema validation rejects malformed model-shaped quiz data.
- JSON extraction handles plain JSON, fenced JSON, extra surrounding text, and missing JSON.
- Empty topics, invalid answer indexes, and invalid review intervals are rejected.
- A wrong quiz answer is promoted into the SQLite-backed review queue.
- A wrong answer is not due until its cooldown advances.
- A due review question reappears after the configured turn interval.
- A correct review answer demotes the item so it is no longer immediately due.
- Session state and review queue state resume from SQLite.
- The review queue returns the highest-priority due item first.
- Frontend-style answer events can grade by `item_id + choice_index`.
- Unknown quiz item IDs are rejected.

## Syntax Check

Command:

```bash
python3 -m compileall quizmaker scripts/run_core_loop.py tests/test_core_loop.py
```

Result:

```text
Listing 'quizmaker'...
Listing 'tests'...
Compiling 'scripts/run_core_loop.py'...
Compiling 'quizmaker/core_loop.py'...
Compiling 'quizmaker/storage.py'...
Compiling 'tests/test_core_loop.py'...
Compiling 'tests/test_gemma_parsing.py'...
Compiling 'tests/test_schemas.py'...
```

Exit code: `0`

## CLI Help Check

Command:

```bash
python3 scripts/run_core_loop.py --help
```

Result:

```text
usage: run_core_loop.py [-h] [--count COUNT] [--review-every REVIEW_EVERY]
                        [--db DB]
                        topic

positional arguments:
  topic                 topic to explain and quiz

options:
  -h, --help            show this help message and exit
  --count COUNT         number of initial MCQs
  --review-every REVIEW_EVERY
                        inject review every K turns
  --db DB               SQLite path for queue and history persistence
```

Exit code: `0`

## Real Gemma Model Run

Command:

```bash
python3 -m uv run scripts/run_core_loop.py "photosynthesis" --count 1 --review-every 1 --db data/real-model-test.sqlite3
```

Result: passed.

The script loaded Gemma, generated a real overview, generated one MCQ, accepted an answer, and graded it correctly.

Sample output:

```text
[Overview]
## Photosynthesis: Quick Overview

* Definition: The process by which plants convert light energy (sunlight) into chemical energy (glucose/sugar).
* Inputs (Reactants): Carbon Dioxide, Water, and Light Energy.
* Location: Occurs primarily in the chloroplasts of plant cells, utilizing the pigment chlorophyll.
* Outputs (Products): Glucose and Oxygen as a byproduct.

[Quiz] What is the primary energy source used by plants during the process of photosynthesis?
  1. Chemical energy stored in glucose
  2. Heat energy from the environment
  3. Light energy from the sun
  4. Energy stored in water molecules
Your answer (1-4): 3
Correct.
Photosynthesis is defined as the process where plants convert light energy (sunlight) into chemical energy (glucose).
```

Notes:

- First non-interactive run generated the overview and MCQ successfully, then stopped at `input()` with `EOFError` because no stdin was attached.
- The interactive rerun completed successfully.
- Hugging Face emitted an unauthenticated-request warning, but model loading and inference completed.
- The generated SQLite test DB lives under `data/`, which is ignored by Git.

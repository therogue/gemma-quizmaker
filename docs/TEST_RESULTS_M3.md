# Test Results — M3

Date: 2026-05-17

Branch: `m3-multi-conversation`

Commit tested: `0f717ba` (`Add optional real model integration test`)

## Environment

- Machine: Apple Silicon iMac (`arm64`, Darwin 25.3.0)
- Python: CPython 3.12.13
- CUDA available: no
- MPS available to PyTorch: no
- Model device path used by `load_model()`: CPU

## Default Test Suite

Command:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 31 tests in 0.281s

OK (skipped=1)
```

The skipped test is `tests.test_real_model_integration`, which is intentionally
disabled unless `RUN_REAL_MODEL_TEST=1` is set.

## Real-Model Integration Test

Command:

```bash
RUN_REAL_MODEL_TEST=1 .venv/bin/python -m unittest tests.test_real_model_integration -v
```

Result:

```text
Ran 1 test in 110.975s

OK
```

The test's measured flow time was:

```text
[real-model integration] completed in 74.0s
```

Observed real model outputs:

- Overview generation returned valid JSON on the first attempt.
- MCQ generation returned valid JSON on the first attempt.
- Chat reply completed successfully.

Coverage exercised:

- `load_model()` with `GemmaQuizGenerator`
- Conversation creation
- Topic overview generation
- MCQ generation
- Answer recording
- Review injection
- Chat reply
- Transcript privacy: question messages do not expose `answer_index` or `rationale`
- Answer reveal after answering
- Graph event persistence to `logs`

## Notes

- This confirms the real model path works on this machine, but it is not a
target GPU performance result because CUDA/MPS were unavailable.
- A target GPU run is still recommended before demo recording to validate
latency on the intended hardware.

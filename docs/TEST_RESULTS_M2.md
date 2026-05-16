# Test Results — M2

Date: 2026-05-10

Branch: `m1-backend-core-loop` (commit `871417b` — merge of `refactor/langgraph-core-loop`)

Tested on: `m1-frontend-html` after rebase onto `m1-backend-core-loop`.

## Environment

- Machine: x86_64 Linux (WSL2)
- OS: Ubuntu on Windows (WSL2 kernel 6.6.87.2)
- Python: CPython 3.12.3
- `uv`: 0.11.9
- GPU: NVIDIA GeForce RTX 3070 Laptop (8 GB VRAM, CUDA available)

## Automated Tests

Command:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
test_answer_item_rejects_unknown_item_id ... ok
test_answer_item_returns_frontend_result_shape ... ok
test_correct_review_demotes_priority ... ok
test_rejects_invalid_configuration_and_inputs ... ok
test_review_queue_returns_highest_priority_due_item ... ok
test_session_and_review_queue_resume_from_sqlite ... ok
test_wrong_answer_is_not_due_until_cooldown_advances ... ok
test_wrong_answer_reappears_as_review_after_interval ... ok
test_extracts_fenced_json ... ok
test_extracts_json_with_extra_text ... ok
test_extracts_plain_json ... ok
test_rejects_output_without_json_object ... ok
test_answer_grading_runs_through_graph_and_logs_grade_node ... ok
test_review_injection_runs_through_graph_and_logs_review_node ... ok
test_safety_failure_refuses_before_content_is_delivered ... ok
test_start_topic_runs_graph_nodes_and_logs_node_inputs_outputs ... ok
test_verify_failure_drops_item_after_one_retry ... ok
test_verify_failure_retries_once_then_accepts_replacement_item ... ok
test_core_loop_exposes_langgraph_runtime_and_m2_nodes ... ok
test_project_declares_langgraph_dependency ... ok
test_accepts_valid_mcq ... ok
test_rejects_empty_required_strings ... ok
test_rejects_invalid_answer_index ... ok
test_rejects_invalid_choice_count ... ok
test_rejects_non_string_choices ... ok

----------------------------------------------------------------------
Ran 25 tests in 0.390s

OK
```

Up from 17 tests at M1. 8 new tests added in `tests/test_langgraph_core_loop.py`.

## New test coverage (M2)

**Structure tests (`LangGraphCoreLoopStructureTests`):**
- `pyproject.toml` declares `langgraph` as a runtime dependency.
- `CoreLoop` exposes `.graph` (LangGraph runtime) and `.graph_node_names` with all 6 M2 nodes.

**Behavior tests (`LangGraphCoreLoopBehaviorTests`):**
- `start_topic` routes through `overview → quiz_gen → verify → safety` and logs all four nodes.
- `verify` node retries once on failure and accepts the replacement item.
- `verify` node drops an item if the replacement also fails.
- `safety` node refuses the topic and returns empty questions before any content is delivered.
- `review_inject` node runs via `next_turn` and logs its output.
- `grade` node runs via `answer` and logs the grading result.

## M1 test coverage (unchanged)

All 8 original `test_core_loop.py` tests continue to pass against the LangGraph-backed `CoreLoop`, confirming behavioral equivalence:

- Wrong answer enters review queue with elevated priority.
- Wrong answer is not due until cooldown advances.
- Due review question reappears after the configured turn interval.
- Correct review answer demotes priority.
- Highest-priority due item surfaces first.
- Session and review queue resume correctly from SQLite.
- Unknown item IDs and invalid inputs are rejected.
- Frontend-style grading by `item_id + choice_index` works.

## Notes

- `verify` and `safety` nodes are wired but use stub implementations (`AcceptAllVerifier`, `AllowAllSafetyChecker`) — real Gemma-backed implementations are the remaining M2 work.
- No real Gemma model run performed; all tests use mock generators.

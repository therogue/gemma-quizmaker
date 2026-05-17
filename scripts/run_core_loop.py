# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "transformers>=4.51.0",
#     "torch",
#     "torchvision",
#     "pillow",
#     "accelerate",
#     "bitsandbytes",
# ]
# ///
"""Run the M1 backend core loop from a terminal.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run scripts/run_core_loop.py "photosynthesis"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quizmaker.core_loop import AskedQuestion, CoreLoop
from quizmaker.gemma import GemmaQuizGenerator, load_model
from quizmaker.storage import QuizStore


def _print_question(asked: AskedQuestion) -> None:
    label = "Review" if asked.is_review else "Quiz"
    print(f"\n[{label}] {asked.mcq.question}")
    for index, choice in enumerate(asked.mcq.choices, start=1):
        print(f"  {index}. {choice}")


def _read_choices() -> list[int]:
    while True:
        raw = input("Your answer (1-4, comma-separated if multiple): ").strip()
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts and all(part in {"1", "2", "3", "4"} for part in parts):
            choices = sorted({int(part) - 1 for part in parts})
            if choices:
                return choices
        print("Enter one or more values from 1, 2, 3, or 4.")


def _ask_and_grade(loop: CoreLoop, conversation_id: int, asked: AskedQuestion) -> None:
    _print_question(asked)
    choices = _read_choices()
    is_correct, rationale = loop.answer(conversation_id, asked, choices)
    correct = ", ".join(str(index + 1) for index in asked.mcq.answer_indices)
    print("Correct." if is_correct else f"Not quite. Correct answer(s): {correct}.")
    print(rationale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="topic to explain and quiz")
    parser.add_argument("--count", type=int, default=3, help="number of initial MCQs")
    parser.add_argument("--review-every", type=int, default=3, help="inject review every K turns")
    parser.add_argument(
        "--db",
        default="data/quizmaker.sqlite3",
        help="SQLite path for queue and history persistence",
    )
    args = parser.parse_args()

    model, processor = load_model()
    store = QuizStore(Path(args.db))
    try:
        loop = CoreLoop(store, GemmaQuizGenerator(model, processor), review_every=args.review_every)
        conversation_id = store.create_conversation()
        overview, questions = loop.start_topic(conversation_id, args.topic, quiz_count=args.count)

        print("\n[Overview]")
        for point in overview.points:
            print(f"  - {point}")

        for asked in questions:
            _ask_and_grade(loop, conversation_id, asked)
            review = loop.next_turn(conversation_id)
            if review is not None:
                _ask_and_grade(loop, conversation_id, review)

        print("\nContinue the session. Press Enter for a turn, or type q to quit.")
        while True:
            raw = input("> ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                break
            review = loop.next_turn(conversation_id)
            if review is None:
                print("No review item due this turn.")
            else:
                _ask_and_grade(loop, conversation_id, review)
    finally:
        store.close()


if __name__ == "__main__":
    main()

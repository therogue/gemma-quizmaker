# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "transformers>=4.51.0",
#     "torch",
#     "torchvision",
#     "pillow",
#     "accelerate",
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


def _read_choice() -> int:
    while True:
        raw = input("Your answer (1-4): ").strip()
        if raw in {"1", "2", "3", "4"}:
            return int(raw) - 1
        print("Enter 1, 2, 3, or 4.")


def _ask_and_grade(loop: CoreLoop, asked: AskedQuestion) -> None:
    _print_question(asked)
    choice = _read_choice()
    is_correct, rationale = loop.answer(asked, choice)
    print("Correct." if is_correct else f"Not quite. Correct answer: {asked.mcq.answer_index + 1}.")
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
        overview, questions = loop.start_topic(args.topic, quiz_count=args.count)

        print("\n[Overview]")
        print(overview)

        for asked in questions:
            _ask_and_grade(loop, asked)
            review = loop.next_turn()
            if review is not None:
                _ask_and_grade(loop, review)

        print("\nContinue the session. Press Enter for a turn, or type q to quit.")
        while True:
            raw = input("> ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                break
            review = loop.next_turn()
            if review is None:
                print("No review item due this turn.")
            else:
                _ask_and_grade(loop, review)
    finally:
        store.close()


if __name__ == "__main__":
    main()

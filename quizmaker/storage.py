"""SQLite persistence for one local M1 learning session."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from quizmaker.schemas import MCQ, ReviewItem


class QuizStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                topic TEXT NOT NULL DEFAULT '',
                overview TEXT NOT NULL DEFAULT '',
                turn_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS quiz_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                choices_json TEXT NOT NULL,
                answer_index INTEGER NOT NULL,
                rationale TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                cooldown INTEGER NOT NULL DEFAULT 0,
                asked_count INTEGER NOT NULL DEFAULT 0,
                correct_count INTEGER NOT NULL DEFAULT 0,
                wrong_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (id, topic, overview, turn_count) VALUES (1, '', '', 0)"
        )
        self.conn.commit()

    def load_session(self) -> dict:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = 1").fetchone()
        return dict(row)

    def save_session(self, topic: str, overview: str, turn_count: int) -> None:
        self.conn.execute(
            """
            UPDATE sessions
            SET topic = ?, overview = ?, turn_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (topic, overview, turn_count),
        )
        self.conn.commit()

    def add_history(self, role: str, kind: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO history (role, kind, content) VALUES (?, ?, ?)",
            (role, kind, content),
        )
        self.conn.commit()

    def add_quiz_item(self, topic: str, mcq: MCQ, priority: int = 0, cooldown: int = 0) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO quiz_items (
                topic, question, choices_json, answer_index, rationale, priority, cooldown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic,
                mcq.question,
                json.dumps(mcq.choices, ensure_ascii=False),
                mcq.answer_index,
                mcq.rationale,
                priority,
                cooldown,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_quiz_items(self, topic: str, mcqs: list[MCQ]) -> list[int]:
        return [self.add_quiz_item(topic, mcq) for mcq in mcqs]

    def get_quiz_item(self, item_id: int) -> ReviewItem | None:
        row = self.conn.execute(
            "SELECT * FROM quiz_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        return self._row_to_review_item(row) if row else None

    def due_review_item(self) -> ReviewItem | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM quiz_items
            WHERE priority > 0 AND cooldown <= 0
            ORDER BY priority DESC, wrong_count DESC, updated_at ASC, id ASC
            LIMIT 1
            """
        ).fetchone()
        return self._row_to_review_item(row) if row else None

    def decrement_cooldowns(self) -> None:
        self.conn.execute(
            """
            UPDATE quiz_items
            SET cooldown = CASE WHEN cooldown > 0 THEN cooldown - 1 ELSE 0 END,
                updated_at = CURRENT_TIMESTAMP
            WHERE priority > 0
            """
        )
        self.conn.commit()

    def record_answer(self, item_id: int, is_correct: bool) -> None:
        if is_correct:
            priority_delta = -1
            cooldown = 4
            correct_inc = 1
            wrong_inc = 0
        else:
            priority_delta = 2
            cooldown = 1
            correct_inc = 0
            wrong_inc = 1

        self.conn.execute(
            """
            UPDATE quiz_items
            SET asked_count = asked_count + 1,
                correct_count = correct_count + ?,
                wrong_count = wrong_count + ?,
                priority = MAX(0, priority + ?),
                cooldown = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (correct_inc, wrong_inc, priority_delta, cooldown, item_id),
        )
        self.conn.commit()

    def mark_wrong_for_review(self, item_id: int) -> None:
        self.conn.execute(
            """
            UPDATE quiz_items
            SET priority = priority + 3,
                cooldown = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (item_id,),
        )
        self.conn.commit()

    def _row_to_review_item(self, row: sqlite3.Row) -> ReviewItem:
        mcq = MCQ.from_mapping(
            {
                "question": row["question"],
                "choices": json.loads(row["choices_json"]),
                "answer_index": row["answer_index"],
                "rationale": row["rationale"],
            }
        )
        return ReviewItem(
            id=row["id"],
            topic=row["topic"],
            mcq=mcq,
            priority=row["priority"],
            cooldown=row["cooldown"],
            asked_count=row["asked_count"],
            correct_count=row["correct_count"],
            wrong_count=row["wrong_count"],
        )

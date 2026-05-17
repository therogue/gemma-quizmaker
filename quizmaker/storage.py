"""SQLite persistence for multi-conversation M3 learning sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from quizmaker.schemas import MCQ, ReviewItem


class QuizStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._detect_old_schema()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _detect_old_schema(self) -> None:
        """Drop DB if stale local-only schemas are detected."""
        if not self.path.exists():
            return
        try:
            conn = sqlite3.connect(self.path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            conn.close()
            if "sessions" in tables and "conversations" not in tables:
                self.path.unlink()
                return
            if "quiz_items" in tables:
                conn = sqlite3.connect(self.path)
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(quiz_items)")
                }
                conn.close()
                if "answer_indices_json" not in columns:
                    self.path.unlink()
        except Exception:
            self.path.unlink(missing_ok=True)

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New conversation',
                topic TEXT NOT NULL DEFAULT '',
                overview_json TEXT NOT NULL DEFAULT '',
                turn_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS quiz_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                choices_json TEXT NOT NULL,
                answer_index INTEGER NOT NULL,
                answer_indices_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                priority INTEGER NOT NULL DEFAULT 0,
                cooldown INTEGER NOT NULL DEFAULT 0,
                asked_count INTEGER NOT NULL DEFAULT 0,
                correct_count INTEGER NOT NULL DEFAULT 0,
                wrong_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_json TEXT NOT NULL,
                quiz_item_id INTEGER REFERENCES quiz_items(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER REFERENCES conversations(id),
                level TEXT NOT NULL DEFAULT 'info',
                event TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    # --- Conversations ---

    def create_conversation(self, title: str = "New conversation") -> int:
        cur = self.conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_conversation(self, conversation_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def save_conversation(
        self,
        conversation_id: int,
        topic: str,
        overview_json: str,
        turn_count: int,
    ) -> None:
        self.conn.execute(
            """
            UPDATE conversations
            SET topic = ?, overview_json = ?, turn_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (topic, overview_json, turn_count, conversation_id),
        )
        self.conn.commit()

    def set_conversation_title(self, conversation_id: int, title: str) -> None:
        self.conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, conversation_id),
        )
        self.conn.commit()

    # --- Quiz items ---

    def add_quiz_item(
        self,
        conversation_id: int,
        topic: str,
        mcq: MCQ,
        priority: int = 0,
        cooldown: int = 0,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO quiz_items (
                conversation_id, topic, question, choices_json,
                answer_index, answer_indices_json, rationale, priority, cooldown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                topic,
                mcq.question,
                json.dumps(mcq.choices, ensure_ascii=False),
                mcq.answer_index,
                json.dumps(mcq.answer_indices, ensure_ascii=False),
                mcq.rationale,
                priority,
                cooldown,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_quiz_items(
        self, conversation_id: int, topic: str, mcqs: list[MCQ]
    ) -> list[int]:
        return [self.add_quiz_item(conversation_id, topic, mcq) for mcq in mcqs]

    def get_quiz_item(self, item_id: int, conversation_id: int) -> ReviewItem | None:
        """Returns None if item doesn't exist or belongs to a different conversation."""
        row = self.conn.execute(
            "SELECT * FROM quiz_items WHERE id = ? AND conversation_id = ?",
            (item_id, conversation_id),
        ).fetchone()
        return self._row_to_review_item(row) if row else None

    def get_active_quiz_items(self, conversation_id: int) -> list[ReviewItem]:
        """All items currently waiting in the user's question panel."""
        rows = self.conn.execute(
            """
            SELECT * FROM quiz_items
            WHERE conversation_id = ? AND status = 'active'
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_review_item(row) for row in rows]

    def get_question_texts(self, conversation_id: int) -> list[str]:
        """All question texts in this conversation, for dedup lookup."""
        return self.list_quiz_questions(conversation_id)

    def list_quiz_questions(
        self, conversation_id: int, topic: str | None = None
    ) -> list[str]:
        """Question text already generated in this conversation."""
        if topic is None:
            rows = self.conn.execute(
                """
                SELECT question FROM quiz_items
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT question FROM quiz_items
                WHERE conversation_id = ? AND topic = ?
                ORDER BY id ASC
                """,
                (conversation_id, topic),
            ).fetchall()
        return [row["question"] for row in rows]

    def deactivate_active_items(self, conversation_id: int) -> None:
        """Retire the current active batch when focus topic changes."""
        self.conn.execute(
            """
            UPDATE quiz_items
            SET status = 'answered', updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ? AND status = 'active'
            """,
            (conversation_id,),
        )
        self.conn.commit()

    def due_review_item(
        self, conversation_id: int, exclude_item_id: int | None = None
    ) -> ReviewItem | None:
        row = self.conn.execute(
            """
            SELECT * FROM quiz_items
            WHERE conversation_id = ?
              AND priority > 0
              AND cooldown <= 0
              AND (? IS NULL OR id != ?)
            ORDER BY priority DESC, wrong_count DESC, updated_at ASC, id ASC
            LIMIT 1
            """,
            (conversation_id, exclude_item_id, exclude_item_id),
        ).fetchone()
        return self._row_to_review_item(row) if row else None

    def decrement_cooldowns(
        self, conversation_id: int, exclude_item_id: int | None = None
    ) -> None:
        self.conn.execute(
            """
            UPDATE quiz_items
            SET cooldown = CASE WHEN cooldown > 0 THEN cooldown - 1 ELSE 0 END,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
              AND priority > 0
              AND (? IS NULL OR id != ?)
            """,
            (conversation_id, exclude_item_id, exclude_item_id),
        )
        self.conn.commit()

    def record_answer(
        self, item_id: int, conversation_id: int, is_correct: bool
    ) -> None:
        if is_correct:
            priority_delta, cooldown, correct_inc, wrong_inc = -1, 4, 1, 0
        else:
            priority_delta, cooldown, correct_inc, wrong_inc = 2, 1, 0, 1

        self.conn.execute(
            """
            UPDATE quiz_items
            SET asked_count = asked_count + 1,
                correct_count = correct_count + ?,
                wrong_count = wrong_count + ?,
                priority = MAX(0, priority + ?),
                cooldown = ?,
                status = 'answered',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND conversation_id = ?
            """,
            (correct_inc, wrong_inc, priority_delta, cooldown, item_id, conversation_id),
        )
        self.conn.commit()

    def activate_for_review(self, item_id: int, conversation_id: int) -> None:
        """Makes a review-scheduled item visible in the question panel again."""
        self.conn.execute(
            """
            UPDATE quiz_items
            SET status = 'active', updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND conversation_id = ?
            """,
            (item_id, conversation_id),
        )
        self.conn.commit()

    def mark_wrong_for_review(self, item_id: int, conversation_id: int) -> None:
        """Re-activates a wrongly answered item so it resurfaces in the question panel."""
        self.conn.execute(
            """
            UPDATE quiz_items
            SET priority = priority + 3,
                cooldown = 1,
                status = 'active',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND conversation_id = ?
            """,
            (item_id, conversation_id),
        )
        self.conn.commit()

    # --- Messages ---

    def add_message(
        self,
        conversation_id: int,
        role: str,
        kind: str,
        content_json: str,
        quiz_item_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO messages (conversation_id, role, kind, content_json, quiz_item_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, kind, content_json, quiz_item_id),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_messages(
        self, conversation_id: int, limit: int | None = None
    ) -> list[dict]:
        if limit is not None:
            rows = self.conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages WHERE conversation_id = ?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Logs ---

    def add_log(
        self,
        event: str,
        content_json: str,
        conversation_id: int | None = None,
        level: str = "info",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO logs (conversation_id, level, event, content_json)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, level, event, content_json),
        )
        self.conn.commit()

    # --- Internal ---

    def _row_to_review_item(self, row: sqlite3.Row) -> ReviewItem:
        mcq = MCQ.from_mapping(
            {
                "question": row["question"],
                "choices": json.loads(row["choices_json"]),
                "answer_indices": json.loads(row["answer_indices_json"]),
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

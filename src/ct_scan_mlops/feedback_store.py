from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

TTL_DAYS = 90

# CT scan classification classes (used for per-class aggregation queries)
_KNOWN_CLASSES = ("adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma", "normal")


class FirestoreFeedbackStore:
    """Feedback store backed by Google Cloud Firestore."""

    def __init__(self, project_id: str | None = None, collection: str = "feedback"):
        from google.cloud import firestore

        self._firestore = firestore
        self.client = firestore.Client(project=project_id)
        self.collection = collection

    def save_feedback(
        self,
        *,
        image_path: str,
        predicted_class: str,
        predicted_confidence: float,
        is_correct: bool,
        correct_class: str | None = None,
        user_note: str | None = None,
        confidence_rating: str | None = None,
        image_stats: dict | None = None,
    ) -> str:
        """Save feedback to Firestore. Returns the document ID."""
        now = datetime.now(UTC)
        doc_data = {
            "timestamp": now.isoformat(),
            "image_path": image_path,
            "predicted_class": predicted_class,
            "predicted_confidence": predicted_confidence,
            "is_correct": is_correct,
            "correct_class": correct_class,
            "user_note": user_note,
            "confidence_rating": confidence_rating,
            "image_stats": image_stats,
            "expireAt": now + timedelta(days=TTL_DAYS),
        }
        _, doc_ref = self.client.collection(self.collection).add(doc_data)
        return doc_ref.id

    def get_stats(self) -> dict:
        """Aggregate feedback statistics using server-side queries.

        Uses Firestore COUNT aggregation for totals and per-class counts,
        and order_by + limit for recent items.  Avoids streaming the entire
        collection into memory.
        """
        col = self.client.collection(self.collection)

        # Server-side COUNT aggregation (O(1) client-side)
        total = col.count(alias="total").get()[0][0].value
        correct = col.where("is_correct", "==", True).count(alias="correct").get()[0][0].value

        # Per-class COUNT queries (one round-trip each)
        class_distribution: dict[str, int] = {}
        for class_name in _KNOWN_CLASSES:
            count = col.where("predicted_class", "==", class_name).count(alias="c").get()[0][0].value
            if count > 0:
                class_distribution[class_name] = count

        # Recent 10 via server-side sort + limit
        recent: list[dict] = []
        for doc in col.order_by("timestamp", direction=self._firestore.Query.DESCENDING).limit(10).stream():
            data = doc.to_dict()
            recent.append(
                {
                    "timestamp": data.get("timestamp"),
                    "predicted_class": data.get("predicted_class", "unknown"),
                    "is_correct": data.get("is_correct", False),
                }
            )

        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "class_distribution": class_distribution,
            "recent_feedback": recent,
            "timestamp": datetime.now(UTC).isoformat(),
        }


class SqliteFeedbackStore:
    """Fallback feedback store using SQLite (for local development)."""

    def __init__(self, db_path: str = "feedback/feedback.db"):
        import sqlite3
        from pathlib import Path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                is_correct BOOLEAN NOT NULL,
                correct_class TEXT,
                user_note TEXT,
                confidence_rating TEXT,
                image_stats TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite feedback database initialized at %s", self.db_path)

    def save_feedback(
        self,
        *,
        image_path: str,
        predicted_class: str,
        predicted_confidence: float,
        is_correct: bool,
        correct_class: str | None = None,
        user_note: str | None = None,
        confidence_rating: str | None = None,
        image_stats: dict | None = None,
    ) -> str:
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO feedback
               (timestamp, image_path, predicted_class, predicted_confidence,
                is_correct, correct_class, user_note, confidence_rating, image_stats)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                image_path,
                predicted_class,
                predicted_confidence,
                is_correct,
                correct_class,
                user_note,
                confidence_rating,
                json.dumps(image_stats) if image_stats else None,
            ),
        )
        conn.commit()
        row_id = str(cursor.lastrowid)
        conn.close()
        return row_id

    def get_stats(self) -> dict:
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1")
        correct_count = cursor.fetchone()[0]

        cursor.execute("SELECT predicted_class, COUNT(*) FROM feedback GROUP BY predicted_class")
        class_distribution = dict(cursor.fetchall())

        cursor.execute("SELECT timestamp, predicted_class, is_correct FROM feedback ORDER BY timestamp DESC LIMIT 10")
        recent_feedback = [
            {"timestamp": row[0], "predicted_class": row[1], "is_correct": bool(row[2])} for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "total_feedback": total_count,
            "correct_predictions": correct_count,
            "incorrect_predictions": total_count - correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "class_distribution": class_distribution,
            "recent_feedback": recent_feedback,
            "timestamp": datetime.now(UTC).isoformat(),
        }


def create_feedback_store() -> FirestoreFeedbackStore | SqliteFeedbackStore:
    """Factory: returns Firestore store if USE_FIRESTORE=1, else SQLite."""
    use_firestore = os.environ.get("USE_FIRESTORE", "0") == "1"
    if use_firestore:
        project_id = os.environ.get("GCP_PROJECT_ID")
        logger.info("Using Firestore feedback store (project=%s)", project_id)
        return FirestoreFeedbackStore(project_id=project_id)
    logger.info("Using SQLite feedback store")
    db_path = os.environ.get("FEEDBACK_DB", "feedback/feedback.db")
    return SqliteFeedbackStore(db_path=db_path)

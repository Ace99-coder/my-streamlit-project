"""
database/database.py
-----------------------
Thin persistence layer over SQLite. Every analysis run is stored so the
Analytics dashboard and Report page can look up past results without
re-running detection, and so Module 8 (multi-document comparison) can
build a similarity matrix from stored results across a session.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

from database.models import CREATE_INDEXES, SCHEMA_STATEMENTS
from utils.helpers import load_config, now_iso, PROJECT_ROOT


class Database:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            cfg = load_config()
            db_path = PROJECT_ROOT / cfg["database"]["path"]
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
            for stmt in CREATE_INDEXES:
                conn.execute(stmt)

    # -- Documents ----------------------------------------------------------

    def insert_document(self, name: str, text: str, upload_date: Optional[str] = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO Documents (document_name, document_text, upload_date) VALUES (?, ?, ?)",
                (name, text, upload_date or now_iso()),
            )
            return cur.lastrowid

    def get_document(self, document_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM Documents WHERE document_id = ?", (document_id,)
            ).fetchone()

    def list_documents(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM Documents ORDER BY upload_date DESC").fetchall()

    # -- AnalysisResults ------------------------------------------------------

    def insert_result(self, **fields: Any) -> int:
        """fields must match AnalysisResults columns (minus result_id)."""
        columns = (
            "suspicious_document_id", "source_document_id", "exact_similarity",
            "tfidf_similarity", "ngram_similarity", "semantic_similarity",
            "detected_plagiarism_type", "classification_confidence",
            "fixed_score", "adaptive_score", "analysis_date",
        )
        fields.setdefault("analysis_date", now_iso())
        values = [fields[c] for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO AnalysisResults ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            return cur.lastrowid

    def list_results_for_document(self, suspicious_document_id: int) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM AnalysisResults WHERE suspicious_document_id = ? "
                "ORDER BY analysis_date DESC",
                (suspicious_document_id,),
            ).fetchall()

    def list_all_results(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM AnalysisResults ORDER BY analysis_date DESC").fetchall()

    # -- SentenceMatches --------------------------------------------------------

    def insert_sentence_matches(self, result_id: int, matches: Sequence[dict]) -> None:
        rows = [
            (
                result_id,
                m["suspicious_sentence"],
                m["source_sentence"],
                m["source_document"],
                m["similarity_score"],
                m["plagiarism_type"],
            )
            for m in matches
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO SentenceMatches "
                "(result_id, suspicious_sentence, source_sentence, source_document, "
                "similarity_score, plagiarism_type) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def list_sentence_matches(self, result_id: int) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM SentenceMatches WHERE result_id = ?", (result_id,)
            ).fetchall()

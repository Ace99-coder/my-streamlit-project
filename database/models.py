"""
database/models.py
---------------------
SQLite schema for the three tables the project spec calls for:
Documents, AnalysisResults, SentenceMatches.

Kept as plain SQL DDL strings (no ORM) to keep the dependency footprint
small — sqlite3 is stdlib, so the whole persistence layer needs nothing
beyond Python itself.
"""

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS Documents (
        document_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name   TEXT NOT NULL,
        document_text   TEXT NOT NULL,
        upload_date     TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS AnalysisResults (
        result_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        suspicious_document_id     INTEGER NOT NULL,
        source_document_id         INTEGER NOT NULL,
        exact_similarity           REAL NOT NULL,
        tfidf_similarity           REAL NOT NULL,
        ngram_similarity           REAL NOT NULL,
        semantic_similarity        REAL NOT NULL,
        detected_plagiarism_type   TEXT NOT NULL,
        classification_confidence  REAL NOT NULL,
        fixed_score                REAL NOT NULL,
        adaptive_score              REAL NOT NULL,
        analysis_date               TEXT NOT NULL,
        FOREIGN KEY (suspicious_document_id) REFERENCES Documents(document_id),
        FOREIGN KEY (source_document_id) REFERENCES Documents(document_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS SentenceMatches (
        match_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id            INTEGER NOT NULL,
        suspicious_sentence  TEXT NOT NULL,
        source_sentence      TEXT NOT NULL,
        source_document      TEXT NOT NULL,
        similarity_score      REAL NOT NULL,
        plagiarism_type       TEXT NOT NULL,
        FOREIGN KEY (result_id) REFERENCES AnalysisResults(result_id)
    );
    """,
]

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_results_suspicious ON AnalysisResults(suspicious_document_id);",
    "CREATE INDEX IF NOT EXISTS idx_matches_result ON SentenceMatches(result_id);",
]

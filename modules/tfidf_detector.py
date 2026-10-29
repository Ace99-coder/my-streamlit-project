"""
modules/tfidf_detector.py
----------------------------
MODULE 3.2: TF-IDF + COSINE SIMILARITY

Connection to the adaptive system:
TF-IDF captures lexical overlap weighted by term rarity — stronger than
exact matching alone because it tolerates word reordering, but still
lexical (not meaning-based). It contributes moderately across almost
every plagiarism type (config.yaml -> adaptive_weights), reflecting that
lexical overlap is informative but never the sole signal.
"""

from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import clamp, load_config, round_score


class TfidfDetector:
    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["detectors"]["tfidf"]
        self.ngram_range = tuple(cfg.get("ngram_range", [1, 2]))
        self.max_features = cfg.get("max_features", 5000)

    def document_score(self, suspicious_text: str, source_text: str) -> float:
        """Whole-document TF-IDF cosine similarity, scaled to 0-100."""
        if not suspicious_text.strip() or not source_text.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )
        try:
            matrix = vectorizer.fit_transform([suspicious_text, source_text])
        except ValueError:
            # happens if, after cleaning, there's no vocabulary left at all
            return 0.0

        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round_score(clamp(sim * 100))

    def sentence_score(self, sentence: str, reference_sentence: str) -> float:
        """Sentence-level TF-IDF similarity, used by Module 6 (sentence
        analysis). Builds a tiny local vocabulary from just the two
        sentences being compared."""
        return self.document_score(sentence, reference_sentence)

    def best_source_index(self, suspicious_text: str, source_texts: List[str]) -> tuple[int, float]:
        """Used by Module 8 (multi-document comparison): finds which
        source document is lexically closest to the suspicious document."""
        if not source_texts:
            return -1, 0.0
        scores = [self.document_score(suspicious_text, s) for s in source_texts]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, scores[best_idx]

"""
modules/exact_matcher.py
--------------------------
MODULE 3.1: EXACT MATCHING

Connection to the adaptive system:
This is the strongest single signal for "Direct Copy" and gets weighted
up to 40% when the classifier detects that pattern (see config.yaml ->
adaptive_weights). It intentionally works on the ORIGINAL sentences
(not the stopword-stripped versions) because verbatim copying is a
surface-level phenomenon — stripping stopwords would hide short exact
copies like "the theory of relativity states that".
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, NamedTuple

from utils.helpers import clamp, round_score


class ExactMatch(NamedTuple):
    suspicious_sentence: str
    source_sentence: str
    ratio: float  # 0-100


class ExactMatcher:
    def __init__(self, config: dict | None = None):
        pass  # no tunables today, but kept for interface symmetry with other detectors

    def sentence_similarity(self, a: str, b: str) -> float:
        """difflib SequenceMatcher ratio, scaled to 0-100."""
        if not a.strip() or not b.strip():
            return 0.0
        ratio = SequenceMatcher(None, a, b).ratio()
        return round_score(clamp(ratio * 100))

    def best_matches(
        self, suspicious_sentences: List[str], source_sentences: List[str]
    ) -> List[ExactMatch]:
        """For each suspicious sentence, find its best exact/near-exact match
        among the source sentences."""
        matches = []
        for s in suspicious_sentences:
            best_ratio, best_source = 0.0, ""
            for src in source_sentences:
                ratio = self.sentence_similarity(s, src)
                if ratio > best_ratio:
                    best_ratio, best_source = ratio, src
            matches.append(ExactMatch(s, best_source, best_ratio))
        return matches

    def document_score(self, suspicious_sentences: List[str], source_sentences: List[str]) -> float:
        """
        Document-level exact-match score = average of each suspicious
        sentence's best match ratio. Using the AVERAGE of best-matches
        (rather than a single whole-document ratio) is what lets this
        detector correctly score documents that copy only *some*
        sentences verbatim (Partial Copy pattern).
        """
        if not suspicious_sentences or not source_sentences:
            return 0.0
        matches = self.best_matches(suspicious_sentences, source_sentences)
        avg = sum(m.ratio for m in matches) / len(matches)
        return round_score(avg)

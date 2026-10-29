"""
modules/ngram_detector.py
----------------------------
MODULE 3.3: N-GRAM SIMILARITY (unigram / bigram / trigram overlap)

Connection to the adaptive system:
N-gram overlap sits between exact matching and TF-IDF: it's more
sensitive to phrase-level copying (word order matters) than TF-IDF, but
more tolerant of minor edits than exact matching. It's the primary
signal for "Structural Similarity" (sentence order/structure preserved)
when combined with a low exact-match score — see plagiarism_classifier.py.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from utils.helpers import clamp, load_config, round_score


def _ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _overlap_coefficient(a: Counter, b: Counter) -> float:
    """
    Overlap coefficient = |A ∩ B| / min(|A|, |B|).
    Chosen over Jaccard because it doesn't penalize a short suspicious
    excerpt being compared against a much longer source document — which
    is common when checking a single paragraph against a full reference.
    """
    if not a or not b:
        return 0.0
    intersection = sum((a & b).values())
    smaller = min(sum(a.values()), sum(b.values()))
    if smaller == 0:
        return 0.0
    return intersection / smaller


class NgramDetector:
    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["detectors"]["ngram"]
        self.sizes = cfg.get("sizes", [1, 2, 3])
        self.weights = cfg.get("weights", [0.2, 0.35, 0.45])
        if len(self.sizes) != len(self.weights):
            raise ValueError("ngram.sizes and ngram.weights must be the same length in config.yaml")

    def document_score(self, suspicious_tokens: List[str], source_tokens: List[str]) -> float:
        """Weighted blend of unigram/bigram/trigram overlap, scaled to 0-100."""
        if not suspicious_tokens or not source_tokens:
            return 0.0

        total = 0.0
        for n, w in zip(self.sizes, self.weights):
            a, b = _ngrams(suspicious_tokens, n), _ngrams(source_tokens, n)
            total += w * _overlap_coefficient(a, b)

        return round_score(clamp(total * 100))

    def sentence_score(self, sentence_tokens: List[str], reference_tokens: List[str]) -> float:
        return self.document_score(sentence_tokens, reference_tokens)

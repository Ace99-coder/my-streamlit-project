"""
modules/hybrid_scorer.py
---------------------------
Computes the Adaptive Plagiarism Score (and, for comparison, the
Traditional Fixed Score) from the four detector scores + a weight vector.

Connection to the adaptive system:
This module is deliberately tiny and dumb — all the "intelligence" lives
upstream in the classifier and adaptive_weighting engine. Keeping the
arithmetic here separate and pure (no config reads, no I/O) makes it
trivially testable (see tests/test_adaptive_weighting.py) and guarantees
the adaptive score and the fixed-comparison score in Module 10 are
computed with the exact same formula, differing only in which weights
are plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from utils.helpers import round_score, weighted_score
from utils.validators import validate_weight_vector


@dataclass
class HybridScoreResult:
    score: float                    # 0-100
    suspected: bool
    weights_used: Dict[str, float]
    detector_scores: Dict[str, float]


class HybridScorer:
    DETECTOR_ORDER = ["exact", "tfidf", "ngram", "semantic"]

    def __init__(self, suspected_threshold: float = 40.0):
        self.suspected_threshold = suspected_threshold

    def compute(
        self,
        detector_scores: Dict[str, float],
        weights: List[float],
        label: str = "weights",
    ) -> HybridScoreResult:
        validate_weight_vector(weights, label=label)
        score = weighted_score(detector_scores, weights)
        return HybridScoreResult(
            score=score,
            suspected=score >= self.suspected_threshold,
            weights_used=dict(zip(self.DETECTOR_ORDER, weights)),
            detector_scores=dict(detector_scores),
        )

    def compare(
        self,
        detector_scores: Dict[str, float],
        adaptive_weights: List[float],
        fixed_weights: List[float],
    ) -> dict:
        """Module 10: Adaptive vs Fixed comparison, in one call."""
        adaptive = self.compute(detector_scores, adaptive_weights, label="adaptive")
        fixed = self.compute(detector_scores, fixed_weights, label="fixed")
        return {
            "adaptive": adaptive,
            "fixed": fixed,
            "difference": round_score(adaptive.score - fixed.score),
        }

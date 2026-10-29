"""
modules/plagiarism_classifier.py
-----------------------------------
MODULE 4: PLAGIARISM TYPE CLASSIFICATION

Connection to the adaptive system:
This is the pivot point of the whole project. It takes the four raw
similarity scores and turns them into a PATTERN LABEL (e.g. "Paraphrased
Plagiarism"). That label is what adaptive_weighting.py uses to pick which
weight vector to apply — nothing downstream (the hybrid score, the
explanation, the report) makes sense without this step running first.

Rules live entirely in config.yaml (`classification_rules`), evaluated
top-to-bottom, first match wins — so the logic is transparent and can be
retuned by editing YAML, not code, satisfying the "configurable, not
hardcoded" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils.helpers import clamp, load_config, round_score


@dataclass
class ClassificationResult:
    plagiarism_type: str
    confidence: float
    matched_rule_index: int
    scores: Dict[str, float]


class PlagiarismClassifier:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.rules: List[Dict[str, Any]] = self.config["classification_rules"]

    # -- public API -----------------------------------------------------

    def classify(self, exact: float, tfidf: float, ngram: float, semantic: float) -> ClassificationResult:
        scores = {"exact": exact, "tfidf": tfidf, "ngram": ngram, "semantic": semantic}

        for idx, rule in enumerate(self.rules):
            conditions = rule.get("conditions", {})
            if self._matches(conditions, scores):
                confidence = self._confidence(rule, conditions, scores)
                return ClassificationResult(
                    plagiarism_type=rule["type"],
                    confidence=confidence,
                    matched_rule_index=idx,
                    scores=scores,
                )

        # Should never happen — config.yaml ends with a conditions={} catch-all —
        # but fail safe rather than raising mid-analysis.
        return ClassificationResult(
            plagiarism_type="Mixed or Complex Similarity",
            confidence=50.0,
            matched_rule_index=-1,
            scores=scores,
        )

    # -- rule evaluation --------------------------------------------------

    @staticmethod
    def _matches(conditions: Dict[str, Any], scores: Dict[str, float]) -> bool:
        """A condition dict like {"exact_gte": 80, "tfidf_gte": 70} means
        ALL keys must hold true (logical AND) for the rule to match."""
        for key, threshold in conditions.items():
            metric, _, op = key.rpartition("_")
            value = scores.get(metric)
            if value is None:
                return False

            if op == "gte" and not (value >= threshold):
                return False
            if op == "lt" and not (value < threshold):
                return False
            if op == "between":
                low, high = threshold
                if not (low <= value <= high):
                    return False
        return True

    @staticmethod
    def _confidence(rule: Dict[str, Any], conditions: Dict[str, Any], scores: Dict[str, float]) -> float:
        """
        Confidence starts at the rule's configured base and is nudged by
        how decisively the scores clear (or barely scrape) their
        thresholds — a document that clears "exact_gte: 80" at 98 is a
        more confident Direct Copy call than one that clears it at 81.
        """
        base = rule.get("confidence_base", 70)
        if not conditions:
            return round_score(clamp(base))

        margins = []
        for key, threshold in conditions.items():
            metric, _, op = key.rpartition("_")
            value = scores.get(metric, 0)

            if op == "gte":
                margin = (value - threshold) / max(1, 100 - threshold)
            elif op == "lt":
                margin = (threshold - value) / max(1, threshold)
            elif op == "between":
                low, high = threshold
                mid = (low + high) / 2
                span = max(1, (high - low) / 2)
                margin = 1 - abs(value - mid) / span
            else:
                margin = 0
            margins.append(margin)

        avg_margin = sum(margins) / len(margins)
        # Each unit of average margin nudges confidence by up to 10 points,
        # keeping the base rating as the dominant factor.
        adjusted = base + (avg_margin * 10)
        return round_score(clamp(adjusted, 0, 99))

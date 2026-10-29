"""
modules/adaptive_weighting.py
--------------------------------
MODULE 5: ADAPTIVE WEIGHTING ENGINE  ⭐ core innovation of the project

Connection to the adaptive system:
Everything before this module (detectors + classifier) exists to feed
this one decision: "given the detected plagiarism TYPE, which detection
method should count the most right now?" Instead of one fixed formula
for every document, this module looks up a weight vector per type from
config.yaml, validates it sums to 1.0, and hands it to hybrid_scorer.py.

This is also what module 10 (Adaptive vs Fixed) is built on: it runs
BOTH this engine's weights AND the fixed baseline weights on the same
scores to prove the adaptive approach isn't just a relabeling exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils.helpers import load_config
from utils.validators import validate_weight_vector


@dataclass
class WeightSelection:
    plagiarism_type: str
    weights: List[float]          # [exact, tfidf, ngram, semantic]
    reason: str
    is_fallback: bool = False


class AdaptiveWeightingEngine:
    DETECTOR_ORDER = ["exact", "tfidf", "ngram", "semantic"]

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.weight_table = self.config["adaptive_weights"]
        # fail fast at startup if config.yaml was edited into an invalid state
        for ptype, entry in self.weight_table.items():
            validate_weight_vector(entry["weights"], label=f"adaptive_weights[{ptype}]")

    def select(self, plagiarism_type: str) -> WeightSelection:
        """Look up the adaptive weight vector for a detected plagiarism type.
        Falls back to equal weighting (25/25/25/25) for any type not
        explicitly configured, so a new classification rule added to
        config.yaml without a matching weight entry degrades safely
        instead of crashing."""
        entry = self.weight_table.get(plagiarism_type)
        if entry is None:
            return WeightSelection(
                plagiarism_type=plagiarism_type,
                weights=[0.25, 0.25, 0.25, 0.25],
                reason=(
                    f"No adaptive weight profile configured for '{plagiarism_type}'; "
                    f"using equal weighting as a safe default."
                ),
                is_fallback=True,
            )
        return WeightSelection(
            plagiarism_type=plagiarism_type,
            weights=list(entry["weights"]),
            reason=entry.get("reason", ""),
        )

    def as_dict(self, selection: WeightSelection) -> dict:
        """Weight vector as a {detector_name: weight} dict — convenient for
        the dashboard's bar chart / radar chart rendering."""
        return dict(zip(self.DETECTOR_ORDER, selection.weights))

    def fixed_weights(self) -> WeightSelection:
        """The traditional, non-adaptive baseline used only for the Module
        10 comparison — never used for the system's own reported score."""
        entry = self.config["fixed_weights"]
        return WeightSelection(
            plagiarism_type="(fixed — all documents)",
            weights=list(entry["weights"]),
            reason=entry.get("label", "Traditional fixed-weight scoring"),
        )

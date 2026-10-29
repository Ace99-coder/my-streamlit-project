"""
modules/explainability_engine.py
------------------------------------
MODULE 7: EXPLAINABLE AI RESULTS

Connection to the adaptive system:
This module writes NOTHING that isn't directly traceable to a value
computed earlier in the pipeline — every sentence in the generated
explanation references a specific detector score, the classification
result, the selected adaptive weights, or a sentence match. That
traceability is the actual "explainability" requirement: it rules out
template text that merely sounds like an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from modules.hybrid_scorer import HybridScoreResult
from modules.plagiarism_classifier import ClassificationResult
from modules.adaptive_weighting import WeightSelection
from modules.sentence_analyzer import SentenceMatch
from utils.helpers import load_config


@dataclass
class Explanation:
    overall_score: float
    plagiarism_type: str
    confidence: float
    suspected: bool
    reasons: List[str]
    conclusion: str


class ExplainabilityEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.high = self.config["scoring"]["high_similarity"]
        self.medium = self.config["scoring"]["medium_similarity"]

    def explain(
        self,
        hybrid_result: HybridScoreResult,
        classification: ClassificationResult,
        weight_selection: WeightSelection,
        sentence_matches: List[SentenceMatch],
    ) -> Explanation:
        reasons: List[str] = []
        scores = classification.scores

        # 1. Lead with whichever detector score is dominant — this is what
        #    actually drove the classification, so it's what the reader
        #    should see first.
        dominant_metric = max(scores, key=scores.get)
        dominant_label = {
            "exact": "Exact text similarity",
            "tfidf": "TF-IDF lexical similarity",
            "ngram": "N-gram phrase overlap",
            "semantic": "Semantic (meaning-level) similarity",
        }[dominant_metric]
        reasons.append(
            f"{dominant_label} was the strongest signal at {scores[dominant_metric]:.1f}%."
        )

        # 2. Contrast against the weakest signal, when it's meaningfully low —
        #    this is exactly the pattern that separates copying from
        #    paraphrasing (e.g. low exact + high semantic).
        weakest_metric = min(scores, key=scores.get)
        if scores[weakest_metric] < self.medium and weakest_metric != dominant_metric:
            weakest_label = {
                "exact": "exact text similarity",
                "tfidf": "TF-IDF lexical similarity",
                "ngram": "n-gram phrase overlap",
                "semantic": "semantic similarity",
            }[weakest_metric]
            reasons.append(
                f"In contrast, {weakest_label} was low at {scores[weakest_metric]:.1f}%, "
                f"which is informative about how the similarity was produced."
            )

        # 3. How many sentences were flagged, and the strongest flagged match.
        flagged = [m for m in sentence_matches if m.is_flagged]
        if flagged:
            top = max(flagged, key=lambda m: m.final_similarity)
            reasons.append(
                f"{len(flagged)} of {len(sentence_matches)} compared sentences were flagged as "
                f"suspicious (similarity ≥ {self.config['scoring']['sentence_flag_threshold']}%). "
                f"The strongest match reached {top.final_similarity:.1f}% similarity with a "
                f"sentence in '{top.source_document}', classified as '{top.classification}'."
            )
        else:
            reasons.append(
                f"No individual sentence exceeded the "
                f"{self.config['scoring']['sentence_flag_threshold']}% flagging threshold."
            )

        # 4. Explicitly connect the classification to the adaptive weight decision.
        reasons.append(
            f"Because the similarity pattern matched '{classification.plagiarism_type}' "
            f"(confidence {classification.confidence:.1f}%), the adaptive scoring engine "
            f"weighted detectors as: Exact {weight_selection.weights[0]*100:.0f}%, "
            f"TF-IDF {weight_selection.weights[1]*100:.0f}%, "
            f"N-Gram {weight_selection.weights[2]*100:.0f}%, "
            f"Semantic {weight_selection.weights[3]*100:.0f}%. {weight_selection.reason}"
        )

        conclusion = self._conclusion(hybrid_result, classification)

        return Explanation(
            overall_score=hybrid_result.score,
            plagiarism_type=classification.plagiarism_type,
            confidence=classification.confidence,
            suspected=hybrid_result.suspected,
            reasons=reasons,
            conclusion=conclusion,
        )

    def _conclusion(self, hybrid_result: HybridScoreResult, classification: ClassificationResult) -> str:
        score = hybrid_result.score
        ptype = classification.plagiarism_type

        if not hybrid_result.suspected:
            return (
                f"The adaptive plagiarism score of {score:.1f}% falls below the suspicion "
                f"threshold. The document does not show a strong pattern of plagiarism "
                f"against the provided reference material."
            )

        if ptype == "Direct Copy":
            return (
                f"The document contains substantial verbatim or near-verbatim text matching "
                f"the reference document(s), consistent with direct copying (score {score:.1f}%)."
            )
        if ptype == "Paraphrased Plagiarism":
            return (
                f"The document contains content that appears to preserve the meaning of the "
                f"reference document while changing the wording (score {score:.1f}%)."
            )
        if ptype == "Structural Similarity":
            return (
                f"The document mirrors the structure and meaning of the reference document "
                f"even where individual wording differs (score {score:.1f}%)."
            )
        if ptype == "Partial Copy":
            return (
                f"Portions of the document closely match the reference material while other "
                f"portions appear original, consistent with partial copying (score {score:.1f}%)."
            )
        return (
            f"The document shows a mixed similarity profile against the reference material "
            f"that does not cleanly fit a single plagiarism pattern (score {score:.1f}%)."
        )

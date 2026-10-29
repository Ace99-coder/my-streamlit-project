"""
modules/sentence_analyzer.py
-------------------------------
MODULE 6: SENTENCE-LEVEL ANALYSIS

Connection to the adaptive system:
Document-level scores tell you THAT something is suspicious; this module
tells you WHERE. For every suspicious sentence it finds the closest
matching source sentence and records exact/TF-IDF/semantic sub-scores,
which is exactly the evidence the explainability engine (Module 7) and
the PDF report (Module 11) quote back to the user as proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from modules.exact_matcher import ExactMatcher
from modules.ngram_detector import NgramDetector
from modules.semantic_detector import SemanticDetector
from modules.tfidf_detector import TfidfDetector
from modules.text_preprocessor import TextPreprocessor
from utils.helpers import load_config, round_score


@dataclass
class SentenceMatch:
    suspicious_sentence: str
    source_sentence: str
    source_document: str
    exact_similarity: float
    tfidf_similarity: float
    semantic_similarity: float
    final_similarity: float
    is_flagged: bool
    classification: str


class SentenceAnalyzer:
    def __init__(
        self,
        exact_matcher: ExactMatcher | None = None,
        tfidf_detector: TfidfDetector | None = None,
        ngram_detector: NgramDetector | None = None,
        semantic_detector: SemanticDetector | None = None,
        preprocessor: TextPreprocessor | None = None,
        config: dict | None = None,
    ):
        self.config = config or load_config()
        self.exact = exact_matcher or ExactMatcher(self.config)
        self.tfidf = tfidf_detector or TfidfDetector(self.config)
        self.ngram = ngram_detector or NgramDetector(self.config)
        self.semantic = semantic_detector or SemanticDetector(self.config)
        self.preprocessor = preprocessor or TextPreprocessor(self.config)
        self.flag_threshold = self.config["scoring"]["sentence_flag_threshold"]
        self.min_sentence_len = self.config["preprocessing"]["min_sentence_length"]

    def analyze(
        self,
        suspicious_sentences: List[str],
        source_sentences: List[str],
        source_document_name: str,
    ) -> List[SentenceMatch]:
        """Compare every suspicious sentence against every source sentence
        for ONE source document, keeping only each suspicious sentence's
        best overall match (by final blended similarity)."""
        results: List[SentenceMatch] = []
        if not suspicious_sentences or not source_sentences:
            return results

        # Pre-tokenize source sentences once for n-gram comparisons.
        source_tokens = [s.lower().split() for s in source_sentences]

        for s_sent in suspicious_sentences:
            if len(s_sent.split()) < self.min_sentence_len:
                continue  # too short to meaningfully compare/flag

            s_tokens = s_sent.lower().split()
            best = None

            for src_sent, src_tok in zip(source_sentences, source_tokens):
                exact = self.exact.sentence_similarity(s_sent, src_sent)
                tfidf = self.tfidf.sentence_score(s_sent, src_sent)
                ngram = self.ngram.sentence_score(s_tokens, src_tok)
                semantic = self.semantic.sentence_similarity(s_sent, src_sent)

                # Simple average across all four for "which source sentence
                # is the overall best match" — the adaptive per-type weights
                # are applied at the DOCUMENT level (hybrid_scorer), not here;
                # this blended figure is only used to pick the best candidate.
                blended = round_score((exact + tfidf + ngram + semantic) / 4)

                if best is None or blended > best["final_similarity"]:
                    best = {
                        "source_sentence": src_sent,
                        "exact": exact,
                        "tfidf": tfidf,
                        "semantic": semantic,
                        "final_similarity": blended,
                    }

            is_flagged = best["final_similarity"] >= self.flag_threshold
            classification = self._classify_sentence(best["exact"], best["semantic"])

            results.append(
                SentenceMatch(
                    suspicious_sentence=s_sent,
                    source_sentence=best["source_sentence"],
                    source_document=source_document_name,
                    exact_similarity=best["exact"],
                    tfidf_similarity=best["tfidf"],
                    semantic_similarity=best["semantic"],
                    final_similarity=best["final_similarity"],
                    is_flagged=is_flagged,
                    classification=classification,
                )
            )

        return results

    @staticmethod
    def _classify_sentence(exact: float, semantic: float) -> str:
        """Lightweight per-sentence label (distinct from the document-level
        PlagiarismClassifier) shown next to each match in the UI/report."""
        if exact >= 80:
            return "Exact / Near-Exact Copy"
        if exact < 40 and semantic >= 75:
            return "Possible Paraphrased Content"
        if semantic >= 60:
            return "Similar Meaning"
        return "Low Similarity"

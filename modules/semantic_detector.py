"""
modules/semantic_detector.py
-------------------------------
MODULE 3.4: SEMANTIC SIMILARITY (Sentence-BERT)

Connection to the adaptive system:
This is the detector that catches paraphrasing, synonym replacement, and
rewritten sentences — the cases where exact/TF-IDF/n-gram all score LOW
because the words changed even though the meaning didn't. It is what
lets the classifier tell "Paraphrased Plagiarism" apart from genuinely
original writing, and the adaptive engine weights it up to 60% for that
pattern (config.yaml -> adaptive_weights).

Offline fallback:
sentence-transformers needs to download `all-MiniLM-L6-v2` from
HuggingFace on first use. In an environment with no internet access,
that download fails. Rather than crashing the whole pipeline, this
module falls back to a character n-gram TF-IDF cosine similarity, which
is a real (if weaker) signal for paraphrase overlap — it still catches
shared roots/substrings across reworded sentences, just not deep meaning
equivalence. The fallback is always logged, never silently pretended to
be BERT-level semantic understanding.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from utils.helpers import clamp, load_config, round_score

logger = logging.getLogger(__name__)


class SemanticDetector:
    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["detectors"]["semantic"]
        self.model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
        self._model = None
        self._using_fallback = False
        self._load_model()

    # -- model loading ------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._using_fallback = False
            logger.info("Loaded Sentence-BERT model '%s'.", self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Could not load Sentence-BERT model '%s' (%s). "
                "Falling back to character n-gram TF-IDF similarity for "
                "the semantic score. Install sentence-transformers and "
                "ensure internet access for full semantic detection.",
                self.model_name,
                exc,
            )
            self._model = None
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    # -- public API -----------------------------------------------------------

    def embed(self, sentences: List[str]) -> np.ndarray:
        """Embed a single batch. Safe to use on its own with the real
        SentenceTransformer model (fixed-dimension output regardless of
        batch), but NOT safe to mix-and-match across two separate calls
        with the fallback vectorizer (see _embed_pair below)."""
        if not sentences:
            return np.zeros((0, 1))
        if self._model is not None:
            return self._model.encode(sentences, show_progress_bar=False)
        return self._fallback_embed(sentences)

    def _embed_pair(self, sentences_a: List[str], sentences_b: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Embed two sentence lists into a SHARED vector space. The real
        SentenceTransformer already guarantees this; the fallback vectorizer
        must be fit jointly on both lists, otherwise the two sets end up
        with different vocabularies/dimensions and cosine similarity breaks."""
        if self._model is not None:
            vecs_a = self._model.encode(sentences_a, show_progress_bar=False) if sentences_a else np.zeros((0, 1))
            vecs_b = self._model.encode(sentences_b, show_progress_bar=False) if sentences_b else np.zeros((0, 1))
            return vecs_a, vecs_b

        combined = self._fallback_embed(sentences_a + sentences_b)
        split = len(sentences_a)
        return combined[:split], combined[split:]

    def sentence_similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        vec_a, vec_b = self._embed_pair([a], [b])
        sim = self._cosine(vec_a[0], vec_b[0])
        return round_score(clamp(sim * 100))

    def document_score(self, suspicious_sentences: List[str], source_sentences: List[str]) -> float:
        """
        Document-level semantic score = average of each suspicious
        sentence's best semantic match among source sentences. Averaging
        best-matches (like exact_matcher does) correctly rewards documents
        where only some sentences are paraphrased from the source.
        """
        if not suspicious_sentences or not source_sentences:
            return 0.0

        susp_vecs, src_vecs = self._embed_pair(suspicious_sentences, source_sentences)

        best_scores = []
        for sv in susp_vecs:
            sims = [self._cosine(sv, tv) for tv in src_vecs]
            best_scores.append(max(sims) if sims else 0.0)

        avg = sum(best_scores) / len(best_scores)
        return round_score(clamp(avg * 100))

    def best_matches(
        self, suspicious_sentences: List[str], source_sentences: List[str]
    ) -> List[tuple[str, str, float]]:
        """Returns (suspicious_sentence, best_source_sentence, similarity%) —
        used directly by Module 6 sentence-level analysis."""
        if not suspicious_sentences or not source_sentences:
            return []

        susp_vecs, src_vecs = self._embed_pair(suspicious_sentences, source_sentences)

        results = []
        for s_sent, sv in zip(suspicious_sentences, susp_vecs):
            sims = [self._cosine(sv, tv) for tv in src_vecs]
            best_i = int(np.argmax(sims)) if sims else -1
            best_sim = round_score(clamp(sims[best_i] * 100)) if best_i >= 0 else 0.0
            best_src = source_sentences[best_i] if best_i >= 0 else ""
            results.append((s_sent, best_src, best_sim))
        return results

    # -- internals --------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _fallback_embed(self, sentences: List[str]) -> np.ndarray:
        """Character 3-5-gram TF-IDF vectors as a stand-in for sentence
        embeddings when the transformer model is unavailable. Fit on the
        batch itself (not a fixed vocabulary) so it works for any input."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return np.zeros((len(sentences), 1))
        return matrix.toarray()

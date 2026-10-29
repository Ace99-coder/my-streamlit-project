"""
modules/text_preprocessor.py
------------------------------
MODULE 2: TEXT PREPROCESSING

Connection to the adaptive system:
TF-IDF and n-gram detectors are sensitive to noise (casing, punctuation,
stopwords) — preprocessing here makes those two detectors comparable to
each other. Exact-match and semantic detectors deliberately work on the
ORIGINAL sentences (preserved separately) because case/punctuation can
matter for exact copying, and Sentence-BERT expects natural sentences.

This module therefore returns a `ProcessedText` object that carries BOTH
representations side by side instead of throwing the original away.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List

from utils.helpers import load_config

# --------------------------------------------------------------------------
# NLTK is the primary backend (as specified in the tech stack). We degrade
# gracefully to lightweight built-in logic if NLTK's data files are not
# available in the current environment (e.g. no internet to fetch them),
# so the pipeline never hard-crashes — it just uses a simpler tokenizer.
# --------------------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    # Probe whether the required corpora are actually downloaded.
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
    _NLTK_READY = True
except Exception:  # pragma: no cover - environment dependent
    _NLTK_READY = False

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "as", "than", "then", "so",
    "such", "not", "no", "nor", "can", "will", "just", "should", "now",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
}


@dataclass
class ProcessedText:
    """Holds both the original and cleaned representations of a document."""

    original_text: str
    original_sentences: List[str] = field(default_factory=list)
    clean_sentences: List[str] = field(default_factory=list)   # lowercased, stopword-free, lemmatized
    tokens: List[str] = field(default_factory=list)             # flat token list of the whole doc, cleaned


class TextPreprocessor:
    def __init__(self, config: dict | None = None):
        self.config = (config or load_config())["preprocessing"]
        self._lemmatizer = WordNetLemmatizer() if _NLTK_READY else None

    # -- public API -------------------------------------------------------

    def process(self, text: str) -> ProcessedText:
        original_sentences = self._sentence_split(text)
        clean_sentences = [self._clean(s) for s in original_sentences]
        # drop sentences that become empty/too-short after cleaning, but
        # keep original_sentences aligned by filtering both in lockstep
        min_len = self.config.get("min_sentence_length", 4)
        paired = [
            (orig, clean)
            for orig, clean in zip(original_sentences, clean_sentences)
            if len(clean.split()) >= 1  # keep short sentences too; min_len only gates flagging later
        ]
        original_sentences = [p[0] for p in paired]
        clean_sentences = [p[1] for p in paired]

        tokens = [tok for sent in clean_sentences for tok in sent.split()]

        return ProcessedText(
            original_text=text,
            original_sentences=original_sentences,
            clean_sentences=clean_sentences,
            tokens=tokens,
        )

    # -- internals ----------------------------------------------------------

    def _sentence_split(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        if _NLTK_READY:
            sentences = sent_tokenize(text)
        else:
            # Simple, dependency-free sentence splitter: split on ./!/? followed
            # by whitespace + capital letter, which covers the vast majority of
            # academic-writing sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean(self, sentence: str) -> str:
        cfg = self.config
        s = sentence

        if cfg.get("lowercase", True):
            s = s.lower()

        if cfg.get("remove_punctuation", True):
            s = s.translate(str.maketrans("", "", string.punctuation))

        s = re.sub(r"\s+", " ", s).strip()

        tokens = word_tokenize(s) if _NLTK_READY else s.split()

        if cfg.get("remove_stopwords", True):
            stop = set(nltk_stopwords.words("english")) if _NLTK_READY else _FALLBACK_STOPWORDS
            tokens = [t for t in tokens if t not in stop]

        if cfg.get("lemmatize", True):
            if self._lemmatizer is not None:
                tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
            else:
                tokens = [self._naive_lemma(t) for t in tokens]

        return " ".join(tokens)

    @staticmethod
    def _naive_lemma(token: str) -> str:
        """Very small suffix-stripping fallback used only when NLTK's
        WordNet corpus isn't available. Not linguistically rigorous, but
        keeps TF-IDF/n-gram overlap calculations reasonable offline."""
        for suffix in ("ing", "edly", "ed", "ies", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

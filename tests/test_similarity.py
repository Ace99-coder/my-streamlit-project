"""
tests/test_similarity.py
---------------------------
Unit tests for the four independent similarity detectors (Module 3).
Each detector is tested in isolation with synthetic sentences chosen to
produce predictable high/low scores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.exact_matcher import ExactMatcher
from modules.ngram_detector import NgramDetector
from modules.semantic_detector import SemanticDetector
from modules.tfidf_detector import TfidfDetector
from utils.helpers import load_config

CONFIG = load_config()


def test_exact_matcher_identical_sentences_scores_100():
    matcher = ExactMatcher(CONFIG)
    score = matcher.sentence_similarity("The quick brown fox jumps.", "The quick brown fox jumps.")
    assert score == 100.0


def test_exact_matcher_unrelated_sentences_scores_low():
    matcher = ExactMatcher(CONFIG)
    score = matcher.sentence_similarity(
        "The quick brown fox jumps over the lazy dog.",
        "Quantum computers use qubits instead of classical bits.",
    )
    assert score < 30.0


def test_exact_matcher_document_score_averages_best_matches():
    matcher = ExactMatcher(CONFIG)
    suspicious = ["Machine learning is powerful.", "The weather is sunny today."]
    source = ["Machine learning is powerful."]
    score = matcher.document_score(suspicious, source)
    # first sentence matches perfectly, second matches poorly -> average is mid-range
    assert 0 < score < 100


def test_tfidf_identical_documents_scores_100():
    detector = TfidfDetector(CONFIG)
    text = "machine learning models require large training datasets"
    score = detector.document_score(text, text)
    assert score == 100.0


def test_tfidf_disjoint_vocabulary_scores_zero():
    detector = TfidfDetector(CONFIG)
    score = detector.document_score("apple banana orange", "rocket satellite orbit")
    assert score == 0.0


def test_ngram_detector_identical_tokens_scores_100():
    detector = NgramDetector(CONFIG)
    tokens = "deep learning uses neural networks".split()
    score = detector.document_score(tokens, tokens)
    assert score == 100.0


def test_ngram_detector_no_overlap_scores_zero():
    detector = NgramDetector(CONFIG)
    a = "deep learning uses neural networks".split()
    b = "stock markets rose sharply today".split()
    score = detector.document_score(a, b)
    assert score == 0.0


def test_ngram_detector_weights_must_match_sizes_length():
    bad_config = {
        **CONFIG,
        "detectors": {
            **CONFIG["detectors"],
            "ngram": {"sizes": [1, 2, 3], "weights": [0.5, 0.5]},
        },
    }
    try:
        NgramDetector(bad_config)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_semantic_detector_identical_sentences_scores_high():
    detector = SemanticDetector(CONFIG)
    score = detector.sentence_similarity(
        "Large datasets can reveal hidden patterns using machine learning.",
        "Large datasets can reveal hidden patterns using machine learning.",
    )
    assert score >= 95.0


def test_semantic_detector_handles_empty_input():
    detector = SemanticDetector(CONFIG)
    assert detector.sentence_similarity("", "something") == 0.0
    assert detector.document_score([], ["a sentence"]) == 0.0


if __name__ == "__main__":
    # Allows `python tests/test_similarity.py` to run without pytest installed.
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

"""
tests/test_classifier.py
---------------------------
Unit tests for Module 4 (PlagiarismClassifier). Each test feeds a
synthetic similarity-score profile matching one plagiarism pattern from
config.yaml and checks the classifier picks the expected type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.plagiarism_classifier import PlagiarismClassifier
from utils.helpers import load_config

CONFIG = load_config()


def test_direct_copy_pattern():
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=95, tfidf=90, ngram=92, semantic=93)
    assert result.plagiarism_type == "Direct Copy"
    assert result.confidence > 90


def test_paraphrased_pattern():
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=15, tfidf=35, ngram=25, semantic=88)
    assert result.plagiarism_type == "Paraphrased Plagiarism"


def test_partial_copy_pattern():
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=45, tfidf=50, ngram=65, semantic=75)
    assert result.plagiarism_type == "Partial Copy"


def test_structural_similarity_pattern():
    # semantic is kept below the Paraphrased-Plagiarism threshold (65) but
    # above the Structural threshold (55) so this profile falls through to
    # Structural Similarity rather than being caught by the Paraphrase rule,
    # which is checked first and does not consider n-gram overlap.
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=20, tfidf=30, ngram=25, semantic=60)
    assert result.plagiarism_type == "Structural Similarity"


def test_low_or_no_plagiarism_pattern():
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=5, tfidf=8, ngram=3, semantic=20)
    assert result.plagiarism_type == "Low or No Significant Plagiarism"


def test_confidence_increases_with_stronger_margin():
    clf = PlagiarismClassifier(CONFIG)
    borderline = clf.classify(exact=81, tfidf=71, ngram=50, semantic=50)
    decisive = clf.classify(exact=99, tfidf=99, ngram=99, semantic=99)
    assert decisive.confidence >= borderline.confidence


def test_classify_returns_all_four_scores():
    clf = PlagiarismClassifier(CONFIG)
    result = clf.classify(exact=10, tfidf=20, ngram=30, semantic=40)
    assert result.scores == {"exact": 10, "tfidf": 20, "ngram": 30, "semantic": 40}


if __name__ == "__main__":
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

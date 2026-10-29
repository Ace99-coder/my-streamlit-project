"""
tests/test_adaptive_weighting.py
------------------------------------
Unit tests for Module 5 (AdaptiveWeightingEngine + HybridScorer) — the
core innovation of the project. Verifies: every configured weight vector
sums to 1.0, the correct vector is selected per plagiarism type, unknown
types degrade safely, and the adaptive/fixed comparison produces a
non-trivial difference on a paraphrase-shaped score profile (proving the
adaptive engine actually changes the outcome, not just relabels it).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.adaptive_weighting import AdaptiveWeightingEngine
from modules.hybrid_scorer import HybridScorer
from utils.helpers import load_config
from utils.validators import ValidationError, validate_weight_vector

CONFIG = load_config()


def test_all_configured_weight_vectors_sum_to_one():
    for ptype, entry in CONFIG["adaptive_weights"].items():
        validate_weight_vector(entry["weights"], label=ptype)  # raises if invalid


def test_fixed_weights_sum_to_one():
    validate_weight_vector(CONFIG["fixed_weights"]["weights"], label="fixed")


def test_invalid_weight_vector_is_rejected():
    try:
        validate_weight_vector([0.5, 0.5, 0.5, 0.5], label="bad")
        raised = False
    except ValidationError:
        raised = True
    assert raised


def test_select_returns_configured_vector_for_known_type():
    engine = AdaptiveWeightingEngine(CONFIG)
    selection = engine.select("Paraphrased Plagiarism")
    assert selection.weights == CONFIG["adaptive_weights"]["Paraphrased Plagiarism"]["weights"]
    assert selection.is_fallback is False


def test_select_falls_back_safely_for_unknown_type():
    engine = AdaptiveWeightingEngine(CONFIG)
    selection = engine.select("Some Unrecognized Type")
    assert selection.weights == [0.25, 0.25, 0.25, 0.25]
    assert selection.is_fallback is True


def test_paraphrase_profile_scores_higher_adaptive_than_fixed():
    """This is the headline claim of the project: for a paraphrase-shaped
    profile (low lexical overlap, high semantic overlap), the adaptive
    engine — which weights semantic similarity heavily for this type —
    should score meaningfully HIGHER than the fixed baseline, which
    under-weights semantic similarity relative to lexical methods."""
    engine = AdaptiveWeightingEngine(CONFIG)
    scorer = HybridScorer(CONFIG["scoring"]["plagiarism_suspected_threshold"])

    scores = {"exact": 15, "tfidf": 25, "ngram": 15, "semantic": 90}
    adaptive_weights = engine.select("Paraphrased Plagiarism").weights
    fixed_weights = engine.fixed_weights().weights

    result = scorer.compare(scores, adaptive_weights, fixed_weights)
    assert result["adaptive"].score > result["fixed"].score
    assert result["difference"] > 0


def test_direct_copy_profile_adaptive_close_to_fixed():
    """For Direct Copy, all four detectors agree (all high), so adaptive
    and fixed weighting should produce SIMILAR scores — the adaptive
    engine's value-add is specifically in cases where detectors disagree."""
    engine = AdaptiveWeightingEngine(CONFIG)
    scorer = HybridScorer(CONFIG["scoring"]["plagiarism_suspected_threshold"])

    scores = {"exact": 92, "tfidf": 90, "ngram": 91, "semantic": 93}
    adaptive_weights = engine.select("Direct Copy").weights
    fixed_weights = engine.fixed_weights().weights

    result = scorer.compare(scores, adaptive_weights, fixed_weights)
    assert abs(result["difference"]) < 5


def test_hybrid_scorer_rejects_invalid_weights():
    scorer = HybridScorer()
    try:
        scorer.compute({"exact": 10, "tfidf": 10, "ngram": 10, "semantic": 10}, [1, 1, 1, 1])
        raised = False
    except ValidationError:
        raised = True
    assert raised


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

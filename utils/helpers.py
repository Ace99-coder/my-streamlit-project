"""
utils/helpers.py
-----------------
Shared, dependency-light helper functions used across the whole system:
config loading (cached), timestamps, score rounding, and file-name safety.

Keeping these in one place means every module reads thresholds/weights
from config.yaml the same way instead of re-implementing YAML loading.
"""

from __future__ import annotations

import functools
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = two levels up from this file (utils/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and cache config.yaml. Cached with lru_cache so every module can
    call load_config() cheaply without re-reading disk on every call.

    Call load_config.cache_clear() in tests if you need to reload after
    editing the file on disk.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. The adaptive engine cannot "
            f"run without its configuration."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a similarity score into [low, high]."""
    return max(low, min(high, value))


def round_score(value: float, ndigits: int = 2) -> float:
    """Round a score for display while keeping it a float (not numpy type)."""
    return round(float(value), ndigits)


def now_iso() -> str:
    """Current timestamp in ISO-8601, used for DB rows and reports."""
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    """Short unique id for in-memory correlation before DB autoincrement is known."""
    return uuid.uuid4().hex[:12]


def safe_filename(name: str) -> str:
    """Strip characters that are unsafe for filenames/paths."""
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name.strip() or "document"


def truncate(text: str, max_len: int = 160) -> str:
    """Truncate long sentence text for compact display in tables/UI."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def weighted_score(scores: Dict[str, float], weights: list[float]) -> float:
    """
    Compute the hybrid score given the four detector scores (in
    [exact, tfidf, ngram, semantic] order) and a matching weight vector.

    Kept here (not just in hybrid_scorer.py) because both the adaptive
    scorer and the fixed-weight comparison (Module 10) need the exact
    same arithmetic to be a fair, apples-to-apples comparison.
    """
    ordered = [scores["exact"], scores["tfidf"], scores["ngram"], scores["semantic"]]
    if len(weights) != 4:
        raise ValueError("weighted_score expects exactly 4 weights.")
    total = sum(w * s for w, s in zip(weights, ordered))
    return round_score(total)

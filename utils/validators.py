"""
utils/validators.py
--------------------
Validation helpers: uploaded file checks and adaptive-weight config
sanity checks (every weight vector must sum to 1.0, or the "hybrid score"
is not actually a weighted average and the whole adaptive claim breaks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class ValidationError(Exception):
    """Raised when uploaded input or configuration fails validation."""


def validate_file_extension(filename: str, allowed: List[str]) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ValidationError(
            f"Unsupported file type '{ext}'. Allowed types: {', '.join(allowed)}"
        )


def validate_file_size(size_bytes: int, max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File is {size_bytes / (1024*1024):.1f}MB, which exceeds the "
            f"{max_mb}MB limit."
        )


def validate_non_empty_text(text: str, source_name: str = "document") -> None:
    if not text or not text.strip():
        raise ValidationError(
            f"No extractable text found in {source_name}. It may be a "
            f"scanned/image-only file or corrupted."
        )


def validate_weight_vector(weights: List[float], label: str, tolerance: float = 1e-6) -> None:
    """A weight vector that doesn't sum to 1.0 silently distorts every score
    it produces, so this is treated as a hard configuration error, not a
    warning — it should fail loudly at startup, not at analysis time."""
    total = sum(weights)
    if len(weights) != 4:
        raise ValidationError(f"Weight vector '{label}' must have exactly 4 values.")
    if abs(total - 1.0) > tolerance:
        raise ValidationError(
            f"Weight vector '{label}' sums to {total:.4f}, not 1.0. "
            f"Check config.yaml -> adaptive_weights / fixed_weights."
        )
    if any(w < 0 for w in weights):
        raise ValidationError(f"Weight vector '{label}' contains a negative weight.")


def validate_config_weights(config: Dict[str, Any]) -> None:
    """Run at app startup: validates every adaptive weight vector plus the
    fixed-weight baseline so a bad config.yaml edit fails fast and visibly
    instead of quietly producing wrong plagiarism scores."""
    for ptype, entry in config.get("adaptive_weights", {}).items():
        validate_weight_vector(entry["weights"], label=f"adaptive_weights[{ptype}]")

    fixed = config.get("fixed_weights", {}).get("weights")
    if fixed:
        validate_weight_vector(fixed, label="fixed_weights")

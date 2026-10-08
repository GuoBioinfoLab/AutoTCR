"""Repertoire-level aggregation of fixed sequence probabilities."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .exceptions import InputFormatError


PRIMARY_METHOD = "ars"
SUPPORTED_METHODS = (
    "ars",
    "top100_mean",
    "top_diff_mean",
    "softmax_sharp",
    "softmax_freq",
)


def normalize_abundance(abundance: Iterable[float]) -> np.ndarray:
    """Convert non-negative clonotype abundance to weights summing to one."""

    values = np.asarray(list(abundance), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise InputFormatError("Abundance must be a non-empty one-dimensional vector.")
    if not np.isfinite(values).all():
        raise InputFormatError("Abundance contains a non-finite value.")
    if (values < 0).any():
        raise InputFormatError("Abundance cannot contain negative values.")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise InputFormatError("Total abundance must be positive.")
    return values / total


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def aggregate_probabilities(
    probabilities: Iterable[float],
    abundance: Optional[Iterable[float]] = None,
    *,
    sharp_temperature: float = 10.0,
) -> Dict[str, float]:
    """Calculate the primary ARS and optional legacy aggregation summaries.

    ``ars`` is the abundance-weighted mean used for the primary repertoire-level analyses. The
    remaining values reproduce exploratory rules from the original research scripts.
    """

    probs = np.asarray(list(probabilities), dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise InputFormatError("Probabilities must be a non-empty one-dimensional vector.")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise InputFormatError("Probabilities must be finite values between 0 and 1.")
    if not np.isfinite(sharp_temperature) or sharp_temperature <= 0:
        raise InputFormatError("sharp_temperature must be positive.")

    freq = np.ones_like(probs) if abundance is None else np.asarray(list(abundance), dtype=float)
    if freq.shape != probs.shape:
        raise InputFormatError("Abundance and probability vectors must have equal length.")
    normalized = normalize_abundance(freq)

    top100 = np.sort(probs)[-min(100, probs.size) :]
    top_k = int(0.1 * probs.size)
    top_diff = 0.0 if top_k == 0 else float(np.sort(probs)[-top_k:].mean() - probs.mean())

    sharp = _stable_softmax(probs * sharp_temperature)
    sharp_freq = normalize_abundance(sharp * normalized)
    return {
        "ars": float(np.dot(normalized, probs)),
        # Scalar class-1 compatibility column, not a stringified [p0, p1] list.
        "freq_prob": float(np.dot(normalized, probs)),
        "top100_mean": float(top100.mean()),
        "top_diff_mean": top_diff,
        "softmax_sharp": float(np.dot(sharp, probs)),
        "softmax_freq": float(np.dot(sharp_freq, probs)),
    }

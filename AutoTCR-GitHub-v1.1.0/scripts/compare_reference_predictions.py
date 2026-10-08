#!/usr/bin/env python3
"""Compare packaged inference with probabilities exported by the original script."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from autotcr import AutoTCRPredictor, InferenceSettings, LegacyTorchBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that packaged AutoTCR probabilities match a frozen reference table."
    )
    parser.add_argument("--settings", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--probability-column", default="autoimmune_probability")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = pd.read_csv(args.reference)
    for column in (args.sequence_column, args.probability_column):
        if column not in reference.columns:
            raise ValueError(f"Reference table is missing required column {column!r}.")

    settings = InferenceSettings.from_file(args.settings)
    predictor = AutoTCRPredictor(
        LegacyTorchBackend(settings),
        positive_class_index=settings.positive_class_index,
    )
    observed = predictor.predict_sequences(
        reference[[args.sequence_column]],
        sequence_column=args.sequence_column,
    )["autoimmune_probability"].to_numpy()
    expected = pd.to_numeric(reference[args.probability_column], errors="raise").to_numpy()
    absolute = np.abs(observed - expected)
    passed = np.allclose(observed, expected, atol=args.atol, rtol=args.rtol)
    print(f"rows={len(reference)}")
    print(f"maximum_absolute_difference={absolute.max():.10g}")
    print(f"mean_absolute_difference={absolute.mean():.10g}")
    print(f"status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())


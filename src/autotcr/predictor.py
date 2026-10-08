"""High-level, user-facing AutoTCR inference interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import logging

import numpy as np
import pandas as pd

from .aggregation import aggregate_probabilities
from .backend import PredictionBackend
from .backend import TorchBackend
from .config import InferenceSettings
from .cohort import resolve_repertoire_path
from .data import prepare_manifest, prepare_sequence_table
from .exceptions import InputFormatError, ModelLoadError
from .utils import ensure_parent


@dataclass
class RepertoirePrediction:
    """Summary metrics and row-level predictions for one repertoire."""

    summary: Dict[str, Any]
    per_sequence: pd.DataFrame

    def save(
        self,
        *,
        summary_path: Optional[Union[str, Path]] = None,
        predictions_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Write either or both result tables as CSV files."""

        if summary_path is not None:
            pd.DataFrame([self.summary]).to_csv(ensure_parent(summary_path), index=False)
        if predictions_path is not None:
            self.per_sequence.to_csv(ensure_parent(predictions_path), index=False)


class AutoTCRPredictor:
    """Stable Python API for sequence, repertoire, and cohort inference."""

    def __init__(self, backend: PredictionBackend, *, positive_class_index: int = 1) -> None:
        if positive_class_index < 0:
            raise ValueError("positive_class_index cannot be negative.")
        self.backend = backend
        self.positive_class_index = positive_class_index

    @classmethod
    def from_pretrained(cls, repo_id: str = "loveCloud/AutoTCR", *, device: str = "auto",
                        batch_size: int = 64, cache_dir: Optional[Union[str, Path]] = None,
                        local_files_only: bool = False) -> "AutoTCRPredictor":
        """Load the pinned public AutoTCR release; reuse cached weights after download.

        The shipped tokenizer and architecture are matched to this specific release.
        Other checkpoints should use from_model_dir or from_settings.
        """
        from .hub import REPO_ID, pretrained_settings
        if repo_id != REPO_ID:
            raise ValueError(f"Bundled assets match {REPO_ID}; use from_model_dir for another model.")
        settings = pretrained_settings(device=device, batch_size=batch_size, cache_dir=cache_dir,
                                       local_files_only=local_files_only)
        return cls(TorchBackend(settings), positive_class_index=settings.positive_class_index)

    @classmethod
    def from_model_dir(cls, model_dir: Union[str, Path], *, device: str = "auto", batch_size: int = 64) -> "AutoTCRPredictor":
        """Load a released model bundle with one call (see models/README.md)."""
        settings = InferenceSettings.from_model_dir(model_dir, device=device, batch_size=batch_size)
        return cls(TorchBackend(settings), positive_class_index=settings.positive_class_index)

    @classmethod
    def from_settings(cls, settings_or_path):
        """Construct from an InferenceSettings object or YAML/JSON path."""
        settings = (settings_or_path if isinstance(settings_or_path, InferenceSettings)
                    else InferenceSettings.from_file(settings_or_path))
        return cls(TorchBackend(settings), positive_class_index=settings.positive_class_index)

    def predict_sequences(
        self,
        table_or_path: Union[pd.DataFrame, str, Path],
        *,
        sequence_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return order-preserving class probabilities for every input sequence."""

        table, seq_col, _ = prepare_sequence_table(
            table_or_path,
            sequence_column=sequence_column,
            require_frequency=False,
        )
        sequences = table[seq_col].tolist()
        probabilities = np.asarray(self.backend.predict_proba(sequences), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(table):
            raise ModelLoadError(
                "Backend probabilities must have shape (number of sequences, number of classes)."
            )
        if self.positive_class_index >= probabilities.shape[1]:
            raise ModelLoadError(
                f"positive_class_index={self.positive_class_index} is invalid for "
                f"{probabilities.shape[1]} output classes."
            )
        if not np.isfinite(probabilities).all():
            raise ModelLoadError("Backend returned a non-finite probability.")
        if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(
            probabilities.sum(axis=1), 1, atol=1e-5, rtol=1e-5
        ):
            raise ModelLoadError("Backend must return normalized class probabilities, not logits.")

        output = table.copy()
        for class_index in range(probabilities.shape[1]):
            output[f"prob_class_{class_index}"] = probabilities[:, class_index]
        output["predicted_label"] = probabilities.argmax(axis=1).astype(int)
        output["autoimmune_probability"] = probabilities[:, self.positive_class_index]
        return output

    def predict_repertoire(
        self,
        table_or_path: Union[pd.DataFrame, str, Path],
        *,
        sample_id: Optional[str] = None,
        sequence_column: Optional[str] = None,
        frequency_column: Optional[str] = None,
        sharp_temperature: float = 10.0,
    ) -> RepertoirePrediction:
        """Predict one repertoire and calculate abundance-weighted ARS."""

        table, seq_col, freq_col = prepare_sequence_table(
            table_or_path,
            sequence_column=sequence_column,
            frequency_column=frequency_column,
            require_frequency=True,
        )
        for column in ("Sample", "sample"):
            if column in table.columns and table[column].dropna().astype(str).nunique() > 1:
                raise InputFormatError("predict_repertoire accepts one repertoire, but multiple sample IDs were found.")
        if sample_id is None:
            for column in ("Sample", "sample"):
                if column in table.columns and table[column].notna().any():
                    sample_id = str(table[column].dropna().iloc[0])
                    break
            if sample_id is None and isinstance(table_or_path, (str, Path)):
                sample_id = Path(table_or_path).name.split(".")[0]
        predictions = self.predict_sequences(table.drop(columns=["input_row"]), sequence_column=seq_col)
        frequencies = predictions[freq_col].to_numpy(dtype=float)
        probabilities = predictions["autoimmune_probability"].to_numpy(dtype=float)
        metrics = aggregate_probabilities(
            probabilities,
            frequencies,
            sharp_temperature=sharp_temperature,
        )
        normalized = frequencies / frequencies.sum()
        predictions["normalized_abundance"] = normalized
        predictions["ars_contribution"] = normalized * probabilities

        summary: Dict[str, Any] = {
            "sample": sample_id,
            "n_clonotypes": int(len(predictions)),
            "n_input_records": int(len(predictions)),
            "n_unique_sequences": int(predictions[seq_col].nunique()),
            "total_abundance": float(frequencies.sum()),
            **metrics,
        }
        return RepertoirePrediction(summary=summary, per_sequence=predictions)

    def predict_cohort(
        self,
        manifest_or_path: Union[pd.DataFrame, str, Path],
        *,
        disease_dir: Optional[Union[str, Path]] = None,
        healthy_dir: Optional[Union[str, Path]] = None,
        sample_column: str = "sample",
        label_column: str = "true_label",
        path_column: str = "file_path",
        file_suffix: str = "_input_seq_vj.csv",
        sequence_column: Optional[str] = None,
        frequency_column: Optional[str] = None,
        manifest_base_dir: Optional[Union[str, Path]] = None,
        skip_missing: bool = False,
        predictions_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Run sample-level inference for every repertoire in a validated manifest."""

        if manifest_base_dir is None and isinstance(manifest_or_path, (str, Path)):
            manifest_base_dir = Path(manifest_or_path).expanduser().resolve().parent
        manifest = prepare_manifest(
            manifest_or_path,
            sample_column=sample_column,
            label_column=label_column,
            path_column=path_column,
        )
        result_rows: List[Dict[str, Any]] = []
        # Cache summaries only; retaining all row-level tables exhausts RAM on large cohorts.
        prediction_cache: Dict[Path, Dict[str, Any]] = {}

        for row_number, (_, row) in enumerate(manifest.iterrows(), start=1):
            logging.getLogger(__name__).info("Sample %d/%d: %s", row_number, len(manifest), row[sample_column])
            path = resolve_repertoire_path(
                row,
                disease_dir=disease_dir,
                healthy_dir=healthy_dir,
                sample_column=sample_column,
                label_column=label_column,
                path_column=path_column,
                file_suffix=file_suffix,
                manifest_base_dir=manifest_base_dir,
            )
            if not path.is_file():
                if skip_missing:
                    output = row.to_dict()
                    for score in ("ars", "freq_prob", "top100_mean", "top_diff_mean", "softmax_sharp", "softmax_freq"):
                        output[score] = np.nan
                    output.update({"repertoire_path": str(path), "status": "missing"})
                    result_rows.append(output)
                    continue
                raise InputFormatError(
                    f"Repertoire file not found for sample {row[sample_column]!r}: {path}"
                )

            if path not in prediction_cache:
                prediction = self.predict_repertoire(
                    path,
                    sample_id=str(row[sample_column]),
                    sequence_column=sequence_column,
                    frequency_column=frequency_column,
                )
                prediction_cache[path] = dict(prediction.summary)
                if predictions_dir is not None:
                    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
                    output_file = ensure_parent(Path(predictions_dir) / f"{digest}.predictions.csv.gz")
                    prediction.per_sequence.to_csv(output_file, index=False)
                    prediction_cache[path]["predictions_file"] = str(output_file)
            summary = dict(prediction_cache[path])
            summary["sample"] = str(row[sample_column])
            output = row.to_dict()
            output.update(summary)
            output.update({"repertoire_path": str(path), "status": "ok"})
            result_rows.append(output)

        return pd.DataFrame(result_rows)


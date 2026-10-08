"""Input-table readers and validation for AutoTCR."""

from __future__ import annotations

from pathlib import Path
import csv
import gzip
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import InputFormatError


SEQUENCE_CANDIDATES = ("sequence", "TcRb_vj", "Sequence", "tcr", "TCR")
FREQUENCY_CANDIDATES = ("Freq", "freq", "frequency", "abundance", "count", "Count")
LABEL_CANDIDATES = ("true_label", "label", "target", "y")


def _find_column(
    columns: Iterable[object],
    requested: Optional[str],
    candidates: Sequence[str],
    *,
    required: bool,
    purpose: str,
) -> Optional[str]:
    names = [str(column) for column in columns]
    lookup = {name.strip().lower(): name for name in names}
    if requested:
        match = lookup.get(str(requested).strip().lower())
        if match is None:
            raise InputFormatError(
                f"Requested {purpose} column {requested!r} was not found. "
                f"Available columns: {names}"
            )
        return match
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise InputFormatError(
            f"Could not identify the {purpose} column. Available columns: {names}. "
            f"Specify it explicitly."
        )
    return None


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read CSV, TSV, TXT, or their gzip-compressed variants with delimiter detection."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise InputFormatError(f"Input table not found: {input_path}")
    try:
        opener = gzip.open if input_path.suffix.lower() == ".gz" else open
        with opener(input_path, "rt", encoding="utf-8-sig") as reader:
            preview = reader.read(4096)
        try:
            delimiter = csv.Sniffer().sniff(preview, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = "\t" if ".tsv" in input_path.name.lower() else ","
        table = pd.read_csv(input_path, sep=delimiter, compression="infer",
                            dtype={"sample": str, "Sample": str}, encoding="utf-8-sig")
    except Exception as exc:
        raise InputFormatError(f"Could not read {input_path}: {exc}") from exc
    if table.empty:
        raise InputFormatError(f"Input table contains no rows: {input_path}")
    return table


def prepare_sequence_table(
    table_or_path: Union[pd.DataFrame, str, Path],
    *,
    sequence_column: Optional[str] = None,
    frequency_column: Optional[str] = None,
    require_frequency: bool = False,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Validate a sequence table and return a normalized copy plus resolved column names."""

    table = (
        read_table(table_or_path)
        if isinstance(table_or_path, (str, Path))
        else table_or_path.copy()
    )
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise InputFormatError("The sequence input must be a non-empty pandas DataFrame or table.")

    seq_col = _find_column(
        table.columns,
        sequence_column,
        SEQUENCE_CANDIDATES,
        required=True,
        purpose="sequence",
    )
    freq_col = _find_column(
        table.columns,
        frequency_column,
        FREQUENCY_CANDIDATES,
        required=require_frequency,
        purpose="frequency",
    )

    if table[seq_col].isna().any():
        rows = table.index[table[seq_col].isna()].tolist()[:5]
        raise InputFormatError(f"Missing sequence values at rows {rows}.")
    # Do not change case or internal spacing: token IDs depend on the supplied string.
    table[seq_col] = table[seq_col].astype(str)
    empty = table[seq_col].str.strip().eq("")
    if empty.any():
        raise InputFormatError(f"Empty sequence values at rows {table.index[empty].tolist()[:5]}.")

    table = table.drop(columns=["input_row"], errors="ignore")
    table.insert(0, "input_row", np.arange(len(table), dtype=np.int64))

    if freq_col is not None:
        numeric = pd.to_numeric(table[freq_col], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            rows = table.index[numeric.isna()].tolist()[:5]
            raise InputFormatError(f"Non-numeric frequency values at rows {rows}.")
        if (numeric < 0).any():
            rows = table.index[numeric < 0].tolist()[:5]
            raise InputFormatError(f"Negative frequency values at rows {rows}.")
        if not np.isfinite(float(numeric.sum())) or float(numeric.sum()) <= 0:
            raise InputFormatError("The sum of repertoire frequencies must be positive.")
        table[freq_col] = numeric.astype(float)

    return table.reset_index(drop=True), seq_col, freq_col


def prepare_manifest(
    table_or_path: Union[pd.DataFrame, str, Path],
    *,
    sample_column: str = "sample",
    label_column: str = "true_label",
    path_column: str = "file_path",
) -> pd.DataFrame:
    """Validate a sample-level manifest without interpreting repertoire paths."""

    table = read_table(table_or_path) if isinstance(table_or_path, (str, Path)) else table_or_path.copy()
    for name in (sample_column,):
        if name not in table.columns:
            raise InputFormatError(
                f"Manifest requires column {name!r}. Available columns: {list(table.columns)}"
            )
    if table[sample_column].isna().any():
        raise InputFormatError("Manifest contains missing sample identifiers.")
    table[sample_column] = table[sample_column].astype(str).str.strip()
    if table[sample_column].eq("").any():
        raise InputFormatError("Manifest contains empty sample identifiers.")
    if label_column not in table.columns:
        if path_column not in table.columns or table[path_column].isna().any() or table[path_column].astype(str).str.strip().eq("").any():
            raise InputFormatError("Unlabelled manifests require file_path for every sample.")
        return table.reset_index(drop=True)
    labels = pd.to_numeric(table[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise InputFormatError(f"Manifest column {label_column!r} must contain only 0 and 1.")
    table[label_column] = labels.astype(int)
    return table.reset_index(drop=True)

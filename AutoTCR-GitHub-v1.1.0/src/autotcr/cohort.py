"""Sample-manifest discovery and repertoire-file resolution."""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Optional, Sequence, Union

import pandas as pd

from .data import prepare_manifest, read_table
from .exceptions import InputFormatError


VALID_EVALUATION_SETS = {"internal", "external"}


def evaluation_set_from_name(path: Union[str, Path]) -> Optional[str]:
    name = Path(path).name.lower()
    if "clinical" in name:
        return None
    if re.search(r"(?:^|_)external(?:_|\s*\.)", name):
        return "external"
    if re.search(r"(?:^|_)internal(?:_|\s*\.)", name):
        return "internal"
    return None


def discover_manifests(
    metadata_dir: Union[str, Path],
    *,
    evaluation_sets: Sequence[str] = ("internal", "external"),
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Combine internal/external metadata CSVs and ignore clinical files."""

    directory = Path(metadata_dir).expanduser().resolve()
    if not directory.is_dir():
        raise InputFormatError(f"Metadata directory not found: {directory}")
    requested = {value.lower() for value in evaluation_sets}
    invalid = requested - VALID_EVALUATION_SETS
    if invalid:
        raise InputFormatError(f"Unknown evaluation sets: {sorted(invalid)}")

    tables: List[pd.DataFrame] = []
    for path in sorted(directory.glob(pattern)):
        evaluation_set = evaluation_set_from_name(path)
        if evaluation_set not in requested:
            continue
        table = read_table(path)
        table["evaluation_set"] = evaluation_set
        table["metadata_file"] = path.name
        table["metadata_base_dir"] = str(path.parent)
        tables.append(table)
    if not tables:
        raise InputFormatError(
            f"No internal/external metadata files matched {pattern!r} in {directory}."
        )
    return prepare_manifest(pd.concat(tables, ignore_index=True, sort=False))


def resolve_repertoire_path(
    row: pd.Series,
    *,
    disease_dir: Optional[Union[str, Path]],
    healthy_dir: Optional[Union[str, Path]],
    sample_column: str = "sample",
    label_column: str = "true_label",
    path_column: str = "file_path",
    file_suffix: str = "_input_seq_vj.csv",
    manifest_base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a repertoire file from an explicit path or the case/control directories."""

    explicit = row.get(path_column)
    if explicit is not None and not pd.isna(explicit) and str(explicit).strip():
        path = Path(str(explicit).strip()).expanduser()
        base = row.get("metadata_base_dir", manifest_base_dir)
        if base is not None and pd.isna(base):
            base = manifest_base_dir
        if not path.is_absolute() and base is not None:
            path = Path(base).expanduser().resolve() / path
        elif not path.is_absolute():
            raise InputFormatError("Relative file_path in a DataFrame requires manifest_base_dir.")
        return path.resolve()

    sample = str(row[sample_column]).strip()
    if sample in {".", ".."} or "/" in sample or "\\" in sample:
        raise InputFormatError("Sample IDs cannot be paths; use file_path for explicit paths.")
    selected = disease_dir if int(row[label_column]) == 1 else healthy_dir
    if selected is None:
        raise InputFormatError("Provide file_path for every sample, or disease_dir and healthy_dir.")
    root = Path(selected)
    return (root.expanduser().resolve() / f"{sample}{file_suffix}").resolve()


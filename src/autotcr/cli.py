"""Command-line interface for AutoTCR inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .backend import TorchBackend
from .cohort import discover_manifests
from .config import InferenceSettings
from .exceptions import AutoTCRError
from .predictor import AutoTCRPredictor
from .utils import ensure_parent

LOGGER = logging.getLogger("autotcr")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--settings",
        default=None,
        help="Explicit YAML/JSON settings. Otherwise use the published Hub release.",
    )
    model_source.add_argument("--model-dir", help="Complete local model directory for offline inference.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face download cache.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached Hub files without network.")
    parser.add_argument("--device", default=None, help="Override auto/cpu/cuda device selection.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotcr",
        description="AutoTCR sequence- and repertoire-level inference",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed progress messages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-model", help="Download a complete offline model bundle.")
    download.add_argument("--output-dir", default="models/autotcr")
    download.add_argument("--cache-dir", default=None)
    download.add_argument("--local-files-only", action="store_true")

    validate = subparsers.add_parser("validate-config", help="Validate paths and import settings.")
    _add_runtime_arguments(validate)

    sequences = subparsers.add_parser("sequences", help="Predict independent receptor sequences.")
    _add_runtime_arguments(sequences)
    sequences.add_argument("--input", required=True, help="Input CSV/TSV table.")
    sequences.add_argument("--output", required=True, help="Output prediction CSV.")
    sequences.add_argument("--sequence-column", default=None)

    repertoire = subparsers.add_parser("repertoire", help="Predict one complete repertoire.")
    _add_runtime_arguments(repertoire)
    repertoire.add_argument("--input", required=True, help="Repertoire CSV/TSV table.")
    repertoire.add_argument("--summary-output", required=True, help="One-row sample summary CSV.")
    repertoire.add_argument(
        "--predictions-output",
        default=None,
        help="Optional row-level prediction CSV or .csv.gz.",
    )
    repertoire.add_argument("--sample-id", default=None)
    repertoire.add_argument("--sequence-column", default=None)
    repertoire.add_argument("--frequency-column", default=None)

    cohort = subparsers.add_parser("cohort", help="Predict all repertoires in a sample manifest.")
    _add_runtime_arguments(cohort)
    source = cohort.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Single sample-level manifest CSV/TSV.")
    source.add_argument(
        "--metadata-dir",
        help="Directory containing *_internal.csv and *_external.csv metadata files.",
    )
    cohort.add_argument("--metadata-pattern", default="*.csv")
    cohort.add_argument(
        "--evaluation-set",
        choices=["internal", "external", "all"],
        default="all",
    )
    cohort.add_argument("--disease-dir", default=None)
    cohort.add_argument("--healthy-dir", default=None)
    cohort.add_argument("--predictions-dir", default=None, help="Optional per-repertoire .csv.gz outputs.")
    cohort.add_argument("--output", required=True)
    cohort.add_argument("--sample-column", default="sample")
    cohort.add_argument("--label-column", default="true_label")
    cohort.add_argument("--path-column", default="file_path")
    cohort.add_argument("--file-suffix", default="_input_seq_vj.csv")
    cohort.add_argument("--sequence-column", default=None)
    cohort.add_argument("--frequency-column", default=None)
    cohort.add_argument("--skip-missing", action="store_true")
    return parser


def _settings_from_args(args: argparse.Namespace) -> InferenceSettings:
    if args.settings is not None:
        settings = InferenceSettings.from_file(args.settings)
    elif args.model_dir is not None:
        settings = InferenceSettings.from_model_dir(args.model_dir)
    else:
        from .hub import pretrained_settings
        settings = pretrained_settings(device=args.device or "auto", batch_size=args.batch_size or 64,
                                       cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    updates = {}
    if args.device is not None:
        updates["device"] = args.device
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if updates:
        settings = replace(settings, **updates)
        settings.validate()
    return settings


def _predictor(settings: InferenceSettings) -> AutoTCRPredictor:
    backend = TorchBackend(settings)
    return AutoTCRPredictor(
        backend,
        positive_class_index=settings.positive_class_index,
    )


def _run(args: argparse.Namespace) -> int:
    if args.command == "download-model":
        from .hub import download_model
        root = download_model(args.output_dir, cache_dir=args.cache_dir,
                              local_files_only=args.local_files_only)
        print(json.dumps({"status": "ok", "model_dir": str(root)}, indent=2))
        return 0
    settings = _settings_from_args(args)

    if args.command == "validate-config":
        settings.validate(check_paths=True)
        backend = TorchBackend(settings)
        print(json.dumps({"status": "ok", **backend.describe()}, indent=2))
        return 0

    predictor = _predictor(settings)
    if args.command == "sequences":
        output = predictor.predict_sequences(args.input, sequence_column=args.sequence_column)
        output_path = ensure_parent(args.output)
        output.to_csv(output_path, index=False)
        LOGGER.info("Wrote %s sequence predictions to %s", len(output), output_path)
        return 0

    if args.command == "repertoire":
        result = predictor.predict_repertoire(
            args.input,
            sample_id=args.sample_id,
            sequence_column=args.sequence_column,
            frequency_column=args.frequency_column,
        )
        result.save(
            summary_path=args.summary_output,
            predictions_path=args.predictions_output,
        )
        print(json.dumps(result.summary, indent=2))
        return 0

    if args.manifest:
        manifest = args.manifest
        manifest_base_dir: Optional[Path] = Path(args.manifest).expanduser().resolve().parent
    else:
        sets = ("internal", "external") if args.evaluation_set == "all" else (args.evaluation_set,)
        manifest = discover_manifests(
            args.metadata_dir,
            evaluation_sets=sets,
            pattern=args.metadata_pattern,
        )
        manifest_base_dir = Path(args.metadata_dir).expanduser().resolve()

    output = predictor.predict_cohort(
        manifest,
        disease_dir=args.disease_dir,
        healthy_dir=args.healthy_dir,
        sample_column=args.sample_column,
        label_column=args.label_column,
        path_column=args.path_column,
        file_suffix=args.file_suffix,
        sequence_column=args.sequence_column,
        frequency_column=args.frequency_column,
        manifest_base_dir=manifest_base_dir,
        skip_missing=args.skip_missing,
        predictions_dir=args.predictions_dir,
    )
    output_path = ensure_parent(args.output)
    output.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s sample predictions to %s", len(output), output_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code for programmatic use."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        return _run(args)
    except (AutoTCRError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())


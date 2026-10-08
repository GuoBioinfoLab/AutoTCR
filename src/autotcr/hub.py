"""Download the verified AutoTCR checkpoint and combine it with bundled assets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from typing import Optional, Union
from .config import InferenceSettings
from .exceptions import ConfigurationError, ModelLoadError

REPO_ID = "loveCloud/AutoTCR"
FILENAME = "AutoTCR.pth"
REVISION = "909bcd38137e73b8a94e709f3758b5d1a8180ebd"
CHECKPOINT_SHA256 = "8949531ceee16f9919b90165d388460c3700d6971958866894ab893076bd58c3"
CHECKPOINT_BYTES = 230268116
ASSET_NAMES = ("config.json", "vocab.txt", "tokenizer_config.json", "inference_config.yaml")


def assets_dir() -> Path:
    """Return the configuration directory shipped in both source and wheel."""
    return Path(__file__).resolve().parent / "assets"


def sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(*, cache_dir=None, local_files_only=False) -> Path:
    """Retrieve the pinned public release and verify its size and SHA-256."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ConfigurationError("Install download/runtime dependencies: pip install -e '.[runtime]'") from exc
    try:
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION,
                                   cache_dir=cache_dir, local_files_only=local_files_only))
    except Exception as exc:
        raise ModelLoadError(
            f"Could not obtain {REPO_ID}/{FILENAME}. "
            "Check network access or download the bundle in advance for offline inference. "
            f"Hub error type: {type(exc).__name__}"
        ) from exc
    if path.stat().st_size != CHECKPOINT_BYTES or sha256(path) != CHECKPOINT_SHA256:
        raise ModelLoadError("Checkpoint integrity check failed for the pinned AutoTCR release.")
    return path.resolve()


def pretrained_settings(*, device="auto", batch_size=64, cache_dir=None,
                        local_files_only=False) -> InferenceSettings:
    """Combine cached Hub weights with the exact author-supplied configuration."""
    assets = assets_dir()
    return InferenceSettings.from_mapping({
        "checkpoint": download_checkpoint(cache_dir=cache_dir, local_files_only=local_files_only),
        "bert_config": assets / "config.json",
        "vocab_file": assets / "vocab.txt",
        "project_config": assets / "inference_config.yaml",
        "device": device, "batch_size": batch_size,
    })


def download_model(output_dir: Union[str, Path], *, cache_dir: Optional[Union[str, Path]] = None,
                   local_files_only: bool = False) -> Path:
    """Write a complete offline bundle; identical assets can be reused.

    Existing files with different contents are never overwritten.
    """
    root = Path(output_dir).expanduser().resolve()
    assets = assets_dir()
    sources = {name: assets / name for name in ASSET_NAMES}
    for name, source in sources.items():
        target = root / name
        if target.exists() and (not target.is_file() or sha256(target) != sha256(source)):
            raise ConfigurationError(f"Different model asset exists: {target}. Use a new directory.")
    weight_target = root / FILENAME
    if weight_target.exists() and (
        not weight_target.is_file() or weight_target.stat().st_size != CHECKPOINT_BYTES
        or sha256(weight_target) != CHECKPOINT_SHA256
    ):
        raise ConfigurationError(f"Different checkpoint exists: {weight_target}. Use a new directory.")
    checkpoint = download_checkpoint(cache_dir=cache_dir, local_files_only=local_files_only)
    sources[FILENAME] = checkpoint
    root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = root / name
        if not target.exists():
            temporary = target.with_name(target.name + ".download.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    metadata = {"repo_id": REPO_ID, "filename": FILENAME, "revision": REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES}
    (root / "model_source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (root / "checksums.json").write_text(
        json.dumps({name: sha256(root / name) for name in sources}, indent=2) + "\n",
        encoding="utf-8")
    return root

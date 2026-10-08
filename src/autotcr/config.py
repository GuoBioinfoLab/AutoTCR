"""Validated, portable settings for the frozen AutoTCR model."""
from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Union
import yaml
from .exceptions import ConfigurationError


def read_config(path: Union[str, Path]) -> dict:
    """Read a YAML/JSON mapping without executing Python or YAML objects."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Invalid configuration {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"Configuration must be a key-value mapping: {path}")
    return dict(data)


@dataclass(frozen=True)
class InferenceSettings:
    """Paths resolve against the settings file, not the current directory.

    project_config supplies the original sent_max_len, dropout and pooling_type.
    Alternatively supply all three explicitly. A full classifier state_dict and
    architecture JSON are sufficient; no separate pretrained weight is needed.
    """
    checkpoint: Path
    bert_config: Path
    vocab_file: Path
    project_config: Optional[Path] = None
    device: str = "auto"
    batch_size: int = 64
    max_length: Optional[int] = None
    dropout: Optional[float] = None
    pooling_type: Optional[str] = None
    positive_class_index: int = 1
    long_sequence_policy: str = "error"
    allow_unsafe_checkpoint: bool = False
    tokenizer_kwargs: Optional[dict] = None

    @classmethod
    def from_file(cls, path):
        """Load portable YAML/JSON settings without loading the model."""
        path = Path(path).expanduser().resolve()
        return cls.from_mapping(read_config(path), base_dir=path.parent)

    @classmethod
    def from_mapping(cls, payload, *, base_dir=None):
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigurationError(f"Unknown inference settings: {', '.join(sorted(unknown))}")
        for key in ("checkpoint", "bert_config", "vocab_file"):
            if not payload.get(key):
                raise ConfigurationError(f"Missing required setting: {key}")
        root = Path(base_dir or ".").expanduser().resolve()
        values = dict(payload)
        for key in ("checkpoint", "bert_config", "vocab_file", "project_config"):
            if values.get(key) is not None:
                path = Path(values[key]).expanduser()
                values[key] = path.resolve() if path.is_absolute() else (root / path).resolve()
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_model_dir(cls, model_dir, **runtime_options):
        """Use AutoTCR.pth (or legacy 10k.pth) with matched local assets."""
        root = Path(model_dir).expanduser().resolve()
        checkpoint_name = "AutoTCR.pth" if (root / "AutoTCR.pth").is_file() else "10k.pth"
        return cls.from_mapping({
            "checkpoint": root / checkpoint_name, "bert_config": root / "config.json",
            "vocab_file": root / "vocab.txt", "project_config": root / "inference_config.yaml",
            **runtime_options,
        })

    def model_options(self):
        """Resolve training-sensitive settings; never guess padding length."""
        original = read_config(self.project_config) if self.project_config else {}
        result = {
            "max_length": self.max_length if self.max_length is not None else original.get("sent_max_len"),
            "dropout": self.dropout if self.dropout is not None else original.get("dropout"),
            "pooling_type": self.pooling_type or original.get("pooling_type"),
            "tokenizer_kwargs": self.tokenizer_kwargs if self.tokenizer_kwargs is not None
                else original.get("tokenizer_kwargs", {}),
        }
        n = result["max_length"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 4:
            raise ConfigurationError("Supply the ORIGINAL sent_max_len (integer >=4) in inference_config.yaml.")
        d = result["dropout"]
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 <= d < 1:
            raise ConfigurationError("Supply the original dropout (0 <= dropout < 1).")
        if result["pooling_type"] not in {"cls", "last-avg", "first-last-avg"}:
            raise ConfigurationError("Use the original pooling_type: cls, last-avg or first-last-avg. "
                                     "The legacy pooler creates an untrained layer inside forward and cannot be reproduced.")
        if not isinstance(result["tokenizer_kwargs"], dict):
            raise ConfigurationError("tokenizer_kwargs must be a mapping.")
        forbidden = {"vocab_file", "model_max_length", "padding_side", "truncation_side"}
        if forbidden & set(result["tokenizer_kwargs"]):
            raise ConfigurationError("tokenizer_kwargs cannot override vocab or padding/truncation policy.")
        return result

    def validate(self, *, check_paths=False):
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("batch_size must be a positive integer.")
        if self.positive_class_index not in (0, 1):
            raise ConfigurationError("positive_class_index must be 0 or 1; the released model is binary.")
        if self.long_sequence_policy not in {"error", "truncate"}:
            raise ConfigurationError("long_sequence_policy must be error or truncate.")
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ConfigurationError("device must be auto, cpu, cuda or cuda:N.")
        if not isinstance(self.allow_unsafe_checkpoint, bool):
            raise ConfigurationError("allow_unsafe_checkpoint must be a boolean.")
        if check_paths:
            for name in ("checkpoint", "bert_config", "vocab_file", "project_config"):
                path = getattr(self, name)
                if path is not None and not Path(path).is_file():
                    raise ConfigurationError(f"Model asset not found ({name}): {path}. See models/README.md.")
            self.model_options()

    def to_dict(self):
        result = asdict(self)
        for key in ("checkpoint", "bert_config", "vocab_file", "project_config"):
            if result[key] is not None:
                result[key] = str(result[key])
        return result

"""Strict offline reconstruction and fixed-length inference for AutoTCR."""
from __future__ import annotations
import logging
import warnings
from typing import Mapping, Protocol, Sequence, runtime_checkable
import numpy as np
from .config import InferenceSettings
from .exceptions import ConfigurationError, ModelLoadError
LOGGER = logging.getLogger(__name__)


@runtime_checkable
class PredictionBackend(Protocol):
    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        """Return an order-matched (n, 2) probability matrix."""


def _load_state_dict(path, torch, *, allow_unsafe=False):
    with open(path, "rb") as reader:
        if reader.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ModelLoadError("Checkpoint is a Git LFS pointer, not weights. Run git lfs pull.")
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe:
            raise ModelLoadError(f"Restricted checkpoint loading failed: {exc}. "
                                 "Only for a trusted checkpoint, allow_unsafe_checkpoint may be enabled.") from exc
        warnings.warn("Unsafe pickle loading enabled: use only your own trusted checkpoint.", RuntimeWarning)
        value = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(value, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(value.get(key), Mapping):
                value = value[key]
                break
    if not isinstance(value, Mapping) or not value or not all(torch.is_tensor(v) for v in value.values()):
        raise ModelLoadError("Expected a full classifier state_dict or a state_dict wrapper.")
    cleaned = {}
    for key, tensor in value.items():
        key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
        if not isinstance(key, str) or key in cleaned:
            raise ModelLoadError("Invalid or colliding keys after removing module. prefixes.")
        cleaned[key] = tensor
    return cleaned


class TorchBackend:
    """Lazy CPU/CUDA backend preserving training-time computation.

    No downloads, dynamic source imports, partial loading or retraining.
    """
    def __init__(self, settings: InferenceSettings):
        settings.validate(check_paths=True)
        self.settings = settings
        self._loaded = False

    @property
    def device(self):
        self._ensure_loaded()
        return str(self._device)

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import torch
            from transformers import AddedToken, BertConfig, BertModel
            from .model import MultiClass
            from .tokenizer import BertTokenizer
        except ImportError as exc:
            raise ConfigurationError("Install inference dependencies: pip install -e '.[runtime]'") from exc
        options = self.settings.model_options()
        name = self.settings.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        if name.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigurationError("CUDA requested but unavailable. Use --device cpu.")
            index = torch.device(name).index
            if index is not None and index >= torch.cuda.device_count():
                raise ConfigurationError(f"CUDA device index {index} is unavailable.")
        try:
            config = BertConfig.from_json_file(str(self.settings.bert_config))
            config._attn_implementation = "eager"
            if config.is_decoder or config.add_cross_attention:
                raise ModelLoadError("The checkpoint requires a BERT encoder, not a decoder.")
            tokenizer_options = dict(options["tokenizer_kwargs"])
            for key in ("unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
                value = tokenizer_options.get(key)
                if isinstance(value, dict):
                    allowed = {"content", "single_word", "lstrip", "rstrip", "normalized", "special"}
                    if "content" not in value:
                        raise ModelLoadError(f"Serialized special token {key} is missing content.")
                    tokenizer_options[key] = AddedToken(**{k: v for k, v in value.items() if k in allowed})
            tokenizer = BertTokenizer(str(self.settings.vocab_file), **tokenizer_options)
            if len(tokenizer) != config.vocab_size:
                raise ModelLoadError(f"Vocabulary size {len(tokenizer)} differs from BERT config {config.vocab_size}.")
            if tokenizer.pad_token_id != 0:
                raise ModelLoadError("Original DataLoader pads with ID 0; the configured pad token must have ID 0.")
            if options["max_length"] > config.max_position_embeddings:
                raise ModelLoadError("sent_max_len exceeds BERT max_position_embeddings.")
            model = MultiClass(BertModel(config), config, options["dropout"],
                               num_classes=2, pooling_type=options["pooling_type"])
            state = _load_state_dict(self.settings.checkpoint, torch,
                                    allow_unsafe=self.settings.allow_unsafe_checkpoint)
            model.load_state_dict(state, strict=True)
        except (ModelLoadError, ConfigurationError):
            raise
        except Exception as exc:
            raise ModelLoadError(f"Could not reconstruct full AutoTCR model: {exc}") from exc
        self._torch = torch
        self._model = model.to(torch.device(name)).eval()
        self._device = torch.device(name)
        self._tokenizer = tokenizer
        self._max_length = options["max_length"]
        self._loaded = True
        LOGGER.info("Loaded full AutoTCR checkpoint on %s (pooling=%s, padded length=%d)",
                    name, options["pooling_type"], self._max_length)

    def describe(self):
        """Return model/runtime metadata after strict loading."""
        self._ensure_loaded()
        return {"device": self.device, "parameters": sum(p.numel() for p in self._model.parameters()),
                "vocab_size": len(self._tokenizer), "max_length": self._max_length,
                "pooling_type": self._model.pooling, "checkpoint": str(self.settings.checkpoint),
                "strict_checkpoint": True}

    def _encode_batch(self, sequences):
        encoded = []
        unknown = 0
        for sequence in sequences:
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError("Each sequence must be a non-empty string.")
            ids = list(self._tokenizer.encode(sequence))
            unknown += int(self._tokenizer.unk_token_id in ids)
            if len(ids) > self._max_length:
                if self.settings.long_sequence_policy == "error":
                    raise ValueError(f"Encoded length {len(ids)} exceeds original sent_max_len={self._max_length}. "
                                     "Verify training settings; changing padding length changes predictions.")
                warnings.warn("Truncation requested; NOT equivalent to the original DataLoader.", RuntimeWarning)
                ids = ids[:self._max_length - 1] + [self._tokenizer.sep_token_id]
            encoded.append(ids)
        if unknown:
            LOGGER.warning("%d/%d sequences contain %s (unknown token); inspect input/vocabulary.",
                           unknown, len(encoded), self._tokenizer.unk_token)
        ids = np.zeros((len(encoded), self._max_length), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, tokens in enumerate(encoded):
            ids[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
        return tuple(self._torch.as_tensor(a, dtype=self._torch.long, device=self._device)
                     for a in (ids, np.zeros_like(ids), mask))

    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        """Predict in FP32 evaluation mode, preserving order."""
        if isinstance(sequences, str):
            raise TypeError("Pass a list of sequences, not a single string.")
        if len(sequences) == 0:
            return np.empty((0, 2), dtype=np.float32)
        self._ensure_loaded()
        batches = []
        with self._torch.inference_mode():
            for start in range(0, len(sequences), self.settings.batch_size):
                inputs = self._encode_batch(sequences[start:start + self.settings.batch_size])
                logits = self._model(*inputs)
                batches.append(self._torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)


# Compatibility name for the previous public interface.
LegacyTorchBackend = TorchBackend

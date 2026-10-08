#!/usr/bin/env python3
"""Copy matched original assets into a portable bundle; no training or downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml
from autotcr.config import InferenceSettings, read_config

TOKENIZER_KEYS = {
    "do_lower_case", "do_basic_tokenize", "never_split", "unk_token",
    "sep_token", "pad_token", "cls_token", "mask_token",
    "tokenize_chinese_chars", "strip_accents", "clean_up_tokenization_spaces",
    "split_special_tokens",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/autotcr"))
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    pretrained = args.pretrained_dir.expanduser().resolve()
    vocab = args.vocab_file.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    sources = {"10k.pth": checkpoint, "config.json": pretrained / "config.json", "vocab.txt": vocab}
    for name, path in sources.items():
        if not path.is_file():
            parser.error(f"Required source missing ({name}): {path}")
    source_config = read_config(args.training_config)
    options = {key: source_config.get(key) for key in ("sent_max_len", "dropout", "pooling_type")}
    tokenizer_options = {}
    tokenizer_config = pretrained / "tokenizer_config.json"
    if tokenizer_config.is_file():
        data = read_config(tokenizer_config)
        tokenizer_options.update({key: data[key] for key in TOKENIZER_KEYS if key in data})
        extras = data.get("added_tokens_decoder", {})
        vocab_size = len(vocab.read_text(encoding="utf-8").splitlines())
        if any(int(index) >= vocab_size for index in extras):
            parser.error("Tokenizer has added tokens outside vocab.txt; supply a matched vocabulary.")
    specials = pretrained / "special_tokens_map.json"
    if specials.is_file():
        tokenizer_options.update({k: v for k, v in read_config(specials).items() if k in TOKENIZER_KEYS})
    override = source_config.get("tokenizer_kwargs", {})
    if not isinstance(override, dict):
        parser.error("tokenizer_kwargs must be a mapping.")
    tokenizer_options.update(override)
    options["tokenizer_kwargs"] = tokenizer_options
    settings = InferenceSettings.from_mapping({
        "checkpoint": checkpoint, "bert_config": sources["config.json"],
        "vocab_file": vocab, "max_length": options["sent_max_len"],
        "dropout": options["dropout"], "pooling_type": options["pooling_type"],
        "tokenizer_kwargs": tokenizer_options,
    })
    settings.model_options()
    targets = list(sources) + ["inference_config.yaml", "checksums.json"]
    if any((output / name).exists() for name in targets):
        parser.error("Destination contains a model asset; use a new directory to avoid overwriting.")
    output.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, output / name)
    (output / "inference_config.yaml").write_text(
        yaml.safe_dump(options, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    checksums = {name: sha256(output / name) for name in targets if name != "checksums.json"}
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared matched model bundle: {output}")
    print("Next: autotcr validate-config --settings configs/inference.yaml --device cpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate the published AutoTCR release against original tokenizer/pooling operations.

Downloads the pinned checkpoint if not cached. This is a software-regression check,
not a biological-performance evaluation. It never writes weights into the GitHub tree.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from autotcr import AutoTCRPredictor
from autotcr.hub import REPO_ID, REVISION, CHECKPOINT_SHA256, assets_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for this validation script.")
    parser.add_argument("--output", default=None, help="Optional JSON validation report.")
    args = parser.parse_args(argv)
    import torch
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    predictor = AutoTCRPredictor.from_pretrained(
        REPO_ID, device=args.device, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only)
    backend = predictor.backend
    metadata = backend.describe()
    original_path = Path(__file__).resolve().parents[1] / "tests/fixtures/original_tokenizer.py"
    spec = importlib.util.spec_from_file_location("original_autotcr_release_tokenizer", original_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assets = assets_dir()
    original = module.BertTokenizer.from_pretrained(str(assets), vocab_file=str(assets / "vocab.txt"))
    texts = [
        "TRBV18CASSSTSDTDTQYFTRBJ2-3",
        "TRBV7-8CASSSSGTTEAFFTRBJ1-1",
        "TRBV10-2CASSFTGGGETQYFTRBJ2-5",
        "TRBV2CATRTGQNQPQHFTRBJ1-5",
    ]
    token_rows, mask_rows = [], []
    for text in texts:
        ids = original.encode(text)
        if ids != backend._tokenizer.encode(text):
            raise AssertionError(f"Original and packaged token IDs differ for {text}")
        token_rows.append(np.pad(ids, (0, 32 - len(ids)), constant_values=0))
        mask_rows.append(np.pad(np.ones(len(ids), dtype=np.int64), (0, 32 - len(ids))))
    tokens = torch.tensor(np.array(token_rows), dtype=torch.long, device=backend._device)
    mask = torch.tensor(np.array(mask_rows), dtype=torch.long, device=backend._device)
    segments = torch.zeros_like(tokens)
    # Independent transcription of the supplied first-last-avg and classification operations.
    with torch.no_grad():
        states = backend._model.bert(tokens, attention_mask=mask, token_type_ids=segments,
                                     output_hidden_states=True)
        first = states.hidden_states[1].transpose(1, 2)
        last = states.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
        pooled = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        x = torch.relu(backend._model.fc1(pooled))
        x = backend._model.dropout(x)
        x = torch.relu(backend._model.fc2(x))
        x = backend._model.dropout(x)
        reference = torch.softmax(backend._model.fc3(x).float(), dim=1).cpu().numpy()
    observed = backend.predict_proba(texts)
    np.testing.assert_allclose(observed, reference, atol=1e-6, rtol=1e-5)
    values = [1.0, 3.0, 2.0, 4.0]
    repertoire = predictor.predict_repertoire(pd.DataFrame({"TcRb_vj": texts, "Freq": values}))
    expected_ars = float(np.dot(np.array(values) / sum(values), reference[:, 1]))
    if abs(repertoire.summary["ars"] - expected_ars) > 1e-6:
        raise AssertionError("ARS differs from the original abundance-weighted formula.")
    report = {
        "status": "PASS", "repo_id": REPO_ID, "revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parameters": metadata["parameters"], "device": metadata["device"],
        "vocab_size": metadata["vocab_size"], "fixed_length": metadata["max_length"],
        "special_token_ids": {key: getattr(backend._tokenizer, key + "_token_id")
                              for key in ("pad", "mask", "unk", "sep", "cls")},
        "reference_sequences": len(texts), "token_ids_match": True,
        "maximum_probability_difference": float(np.abs(observed - reference).max()),
        "ars": repertoire.summary["ars"], "ars_difference": abs(repertoire.summary["ars"] - expected_ars),
        "scope": "Original tokenizer and pooling operations; shared loaded encoder/head parameters."
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CPU integration and legacy-computation checks using a tiny random checkpoint."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
import yaml

RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(RUNTIME, "Install runtime dependencies to run CPU model tests.")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertConfig, BertModel
        from autotcr.model import MultiClass
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.torch = torch
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "TRBV18", "TRBV7-8",
                 "TRBJ2-3", "TRBJ1-1"] + list("ACDEFGHIKLMNPQRSTVWY")
        (cls.root / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=32,
                         max_position_embeddings=64, hidden_dropout_prob=0,
                         attention_probs_dropout_prob=0)
        cfg._attn_implementation = "eager"
        cfg.to_json_file(cls.root / "config.json")
        cls.model = MultiClass(BertModel(cfg), cfg, 0.2).eval()
        torch.save(cls.model.state_dict(), cls.root / "10k.pth")
        (cls.root / "inference_config.yaml").write_text(
            yaml.safe_dump({"sent_max_len": 32, "dropout": 0.2,
                            "pooling_type": "first-last-avg"}), encoding="utf-8")
        from autotcr import InferenceSettings, TorchBackend
        cls.settings = InferenceSettings.from_model_dir(cls.root, device="cpu", batch_size=2)
        cls.backend = TorchBackend(cls.settings)
        cls.backend.describe()
        cls.sequences = ["TRBV18CASSSTSDTDTQYFTRBJ2-3", "TRBV7-8CASSTRBJ1-1"]
        spec = importlib.util.spec_from_file_location(
            "original_autotcr_tokenizer", Path(__file__).parent / "fixtures" / "original_tokenizer.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        cls.original_tokenizer = original.BertTokenizer(str(cls.root / "vocab.txt"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def backend_for(self, **kwargs):
        from autotcr import TorchBackend
        return TorchBackend(replace(self.settings, **kwargs))

    def test_original_tokenization_and_ids(self):
        texts = self.sequences + ["TRBV18 CASS TRBJ2-3", "unknown", "A" * 101, "[MASK]"]
        for text in texts:
            self.assertEqual(self.original_tokenizer.tokenize(text), self.backend._tokenizer.tokenize(text))
            self.assertEqual(self.original_tokenizer.encode(text), self.backend._tokenizer.encode(text))

    def test_fixed_padding_matches_original_loader(self):
        observed = self.backend._encode_batch(self.sequences)
        tokens, masks = [], []
        for seq in self.sequences:
            ids = self.original_tokenizer.encode(seq)
            tokens.append(np.pad(ids, (0, 32 - len(ids)), constant_values=0))
            masks.append(np.pad(np.ones(len(ids), dtype=int), (0, 32 - len(ids)), constant_values=0))
        np.testing.assert_array_equal(observed[0].cpu().numpy(), tokens)
        np.testing.assert_array_equal(observed[1].cpu().numpy(), np.zeros((2, 32), dtype=int))
        np.testing.assert_array_equal(observed[2].cpu().numpy(), masks)

    def test_original_pooling_and_head_computation(self):
        torch = self.torch
        tokens, segments, masks = self.backend._encode_batch(self.sequences)
        with torch.no_grad():
            states = self.model.bert(tokens, attention_mask=masks, token_type_ids=segments,
                                     output_hidden_states=True)
            for mode in ("cls", "last-avg", "first-last-avg"):
                if mode == "cls":
                    pooled = states.last_hidden_state[:, 0, :]
                elif mode == "last-avg":
                    last = states.last_hidden_state.transpose(1, 2)
                    pooled = torch.avg_pool1d(last, last.shape[-1]).squeeze(-1)
                else:
                    first = states.hidden_states[1].transpose(1, 2)
                    last = states.hidden_states[-1].transpose(1, 2)
                    first_avg = torch.avg_pool1d(first, last.shape[-1]).squeeze(-1)
                    last_avg = torch.avg_pool1d(last, last.shape[-1]).squeeze(-1)
                    avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
                    pooled = torch.avg_pool1d(avg.transpose(1, 2), 2).squeeze(-1)
                expected = self.model.fc3(torch.relu(self.model.fc2(torch.relu(self.model.fc1(pooled)))))
                backend = self.backend_for(pooling_type=mode)
                backend.describe()
                np.testing.assert_allclose(backend._model(tokens, segments, masks).numpy(),
                                           expected.numpy(), atol=1e-7, rtol=1e-6)

    def test_probabilities_and_batch_size(self):
        expected = self.backend.predict_proba(self.sequences)
        observed = self.backend_for(batch_size=1).predict_proba(self.sequences)
        np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-7)

    def test_module_prefix_and_checkpoint_wrapper(self):
        path = self.root / "wrapped.pth"
        self.torch.save({"state_dict": {"module." + k: v for k, v in self.model.state_dict().items()}}, path)
        np.testing.assert_allclose(self.backend_for(checkpoint=path).predict_proba(self.sequences),
                                   self.backend.predict_proba(self.sequences), atol=1e-7)

    def test_partial_checkpoint_is_rejected(self):
        from autotcr.exceptions import ModelLoadError
        path = self.root / "partial.pth"
        state = dict(self.model.state_dict())
        del state["fc3.bias"]
        self.torch.save(state, path)
        with self.assertRaises(ModelLoadError):
            self.backend_for(checkpoint=path).describe()

    def test_lfs_pointer_is_rejected(self):
        from autotcr.exceptions import ModelLoadError
        path = self.root / "pointer.pth"
        path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        with self.assertRaisesRegex(ModelLoadError, "git lfs pull"):
            self.backend_for(checkpoint=path).describe()

    def test_vocabulary_mismatch_is_rejected(self):
        from autotcr.exceptions import ModelLoadError
        path = self.root / "bad_vocab.txt"
        path.write_text((self.root / "vocab.txt").read_text() + "NEW\n", encoding="utf-8")
        with self.assertRaises(ModelLoadError):
            self.backend_for(vocab_file=path).describe()

    def test_serialized_special_tokens_preserve_ids(self):
        backend = self.backend_for(tokenizer_kwargs={
            "cls_token": {"content": "[CLS]", "special": True, "normalized": False}})
        np.testing.assert_allclose(backend.predict_proba(self.sequences),
                                   self.backend.predict_proba(self.sequences), atol=1e-7)

    def test_overlength_input_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "sent_max_len"):
            self.backend.predict_proba(["A" * 40])

    def test_string_instead_of_sequence_list_is_rejected(self):
        with self.assertRaises(TypeError):
            self.backend.predict_proba(self.sequences[0])

    def test_repertoire_order_and_ars(self):
        from autotcr import AutoTCRPredictor
        table = pd.DataFrame({"TcRb_vj": self.sequences, "Freq": [1, 3]})
        result = AutoTCRPredictor(self.backend).predict_repertoire(table, sample_id="example")
        expected = np.dot([0.25, 0.75], self.backend.predict_proba(self.sequences)[:, 1])
        self.assertAlmostEqual(result.summary["ars"], expected)
        self.assertEqual(result.summary["ars"], result.summary["freq_prob"])
        self.assertEqual(result.per_sequence["TcRb_vj"].tolist(), self.sequences)
        self.assertAlmostEqual(result.per_sequence["ars_contribution"].sum(), expected)

    def run_cli(self, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        result = subprocess.run([sys.executable, "-m", "autotcr", *map(str, args)],
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_cpu_cli_end_to_end(self):
        settings = self.root / "settings.yaml"
        settings.write_text(yaml.safe_dump(self.settings.to_dict()))
        repertoire = self.root / "repertoire.csv"
        pd.DataFrame({"TcRb_vj": self.sequences, "Freq": [1, 3]}).to_csv(repertoire, index=False)
        sequences_out = self.root / "sequence_output.csv"
        summary_out = self.root / "summary.csv"
        self.run_cli("validate-config", "--settings", settings, "--device", "cpu")
        self.run_cli("sequences", "--settings", settings, "--input", repertoire, "--output", sequences_out)
        self.run_cli("repertoire", "--settings", settings, "--input", repertoire,
                     "--summary-output", summary_out)
        manifest = self.root / "manifest.csv"
        pd.DataFrame({"sample": ["unlabelled"], "file_path": ["repertoire.csv"]}).to_csv(manifest, index=False)
        output = self.root / "cohort.csv"
        self.run_cli("cohort", "--settings", settings, "--manifest", manifest, "--output", output)
        self.assertEqual(pd.read_csv(output)["status"].iloc[0], "ok")
        self.assertEqual(len(pd.read_csv(sequences_out)), 2)
        self.assertAlmostEqual(pd.read_csv(output)["ars"].iloc[0], pd.read_csv(summary_out)["ars"].iloc[0])

    def test_model_bundle_exporter(self):
        repo = Path(__file__).resolve().parents[1]
        output = self.root / "exported"
        args = [sys.executable, str(repo / "scripts" / "prepare_model_bundle.py"),
                "--checkpoint", str(self.root / "10k.pth"),
                "--pretrained-dir", str(self.root), "--vocab-file", str(self.root / "vocab.txt"),
                "--training-config", str(self.root / "inference_config.yaml"),
                "--output-dir", str(output)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((output / "checksums.json").read_text())), 4)
        from autotcr import AutoTCRPredictor
        prediction = AutoTCRPredictor.from_model_dir(output, device="cpu").predict_sequences(
            pd.DataFrame({"sequence": self.sequences}))
        np.testing.assert_allclose(prediction["prob_class_1"], self.backend.predict_proba(self.sequences)[:, 1])
        again = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(again.returncode, 0)

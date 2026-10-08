"""Test shipped release metadata and download plumbing without network or real weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from autotcr.config import InferenceSettings, read_config
from autotcr.hub import assets_dir, download_model, pretrained_settings, REPO_ID
from autotcr.exceptions import ConfigurationError


class HubTests(unittest.TestCase):
    def test_real_vocabulary_order_and_options(self):
        assets = assets_dir()
        tokens = (assets / "vocab.txt").read_text().splitlines()
        self.assertEqual(len(tokens), 103)
        self.assertEqual(tokens[:5], ["$", ".", "?", "|", "*"])
        settings = InferenceSettings.from_mapping({
            "checkpoint": "unused.pth", "bert_config": assets / "config.json",
            "vocab_file": assets / "vocab.txt", "project_config": assets / "inference_config.yaml"})
        options = settings.model_options()
        self.assertEqual(options["max_length"], 32)
        self.assertEqual(options["dropout"], 0.3)
        self.assertEqual(options["tokenizer_kwargs"]["cls_token"], "*")

    def test_pinned_settings_use_bundled_assets(self):
        with patch("autotcr.hub.download_checkpoint", return_value=Path("/tmp/mock.pth")):
            settings = pretrained_settings(device="cpu")
        self.assertEqual(settings.checkpoint, Path("/tmp/mock.pth"))
        self.assertEqual(settings.bert_config, assets_dir() / "config.json")
        self.assertEqual(settings.batch_size, 64)

    def test_download_bundle_and_reject_different_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "mock.pth"
            checkpoint.write_bytes(b"synthetic test file")
            output = root / "bundle"
            with patch("autotcr.hub.download_checkpoint", return_value=checkpoint):
                download_model(output)
            self.assertEqual((output / "AutoTCR.pth").read_bytes(), checkpoint.read_bytes())
            self.assertEqual(read_config(output / "model_source.json")["repo_id"], REPO_ID)
            (output / "vocab.txt").write_text("different\n")
            with self.assertRaises(ConfigurationError):
                download_model(output)

    def test_local_release_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AutoTCR.pth").touch()
            settings = InferenceSettings.from_model_dir(root)
            self.assertEqual(settings.checkpoint.name, "AutoTCR.pth")

    def test_wrong_repository_is_explicit(self):
        from autotcr import AutoTCRPredictor
        with self.assertRaises(ValueError):
            AutoTCRPredictor.from_pretrained("someone/another-model")

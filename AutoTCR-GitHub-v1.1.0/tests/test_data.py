import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotcr.config import InferenceSettings
from autotcr.data import prepare_sequence_table
from autotcr.exceptions import ConfigurationError, InputFormatError


class DataTests(unittest.TestCase):
    def test_sequence_and_frequency_detection(self):
        table = pd.DataFrame({"TcRb_vj": ["TRBV1CASSTRBJ1-1"], "Freq": [3]})
        normalized, seq_col, freq_col = prepare_sequence_table(
            table,
            require_frequency=True,
        )
        self.assertEqual(seq_col, "TcRb_vj")
        self.assertEqual(freq_col, "Freq")
        self.assertEqual(normalized.loc[0, "input_row"], 0)

    def test_negative_frequency_is_rejected(self):
        table = pd.DataFrame({"sequence": ["ABC"], "Freq": [-1]})
        with self.assertRaises(InputFormatError):
            prepare_sequence_table(table, require_frequency=True)

    def test_settings_resolve_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "project_config": "config.json",
                "checkpoint": "model.pth",
                "vocab_file": "vocab.txt",
                "bert_config": "bert_config.json",
            }
            settings = InferenceSettings.from_mapping(payload, base_dir=root)
            self.assertEqual(settings.checkpoint, (root / "model.pth").resolve())

    def test_unknown_settings_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            InferenceSettings.from_mapping(
                {
                    "project_config": "a",
                    "checkpoint": "b",
                    "vocab_file": "c",
                    "mystery": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()


"""Additional public-input checks independent of PyTorch."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from autotcr.data import prepare_manifest, prepare_sequence_table, read_table
from autotcr.cohort import evaluation_set_from_name
from autotcr.config import InferenceSettings
from autotcr.exceptions import ConfigurationError, InputFormatError
from autotcr.predictor import AutoTCRPredictor


class InputsTests(unittest.TestCase):
    def test_one_column_csv_and_tab_delimited_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seq.csv"
            path.write_text("sequence\nABC\nDEF\n")
            self.assertEqual(read_table(path)["sequence"].tolist(), ["ABC", "DEF"])
            path.write_text("sequence\ttrue_label\nABC\t1\n")
            self.assertEqual(read_table(path)["true_label"].tolist(), [1])

    def test_duplicate_rows_case_and_order_preserved(self):
        table = pd.DataFrame({"sequence": ["AbC", "AbC", "DEF"], "Freq": [1, 2, 3]})
        observed, _, _ = prepare_sequence_table(table)
        self.assertEqual(observed["sequence"].tolist(), ["AbC", "AbC", "DEF"])
        self.assertEqual(observed["input_row"].tolist(), [0, 1, 2])

    def test_nonfinite_and_zero_total_abundances(self):
        for values in ([np.nan], [np.inf], [0], [-1]):
            with self.assertRaises(InputFormatError):
                prepare_sequence_table(pd.DataFrame({"sequence": ["ABC"], "Freq": values}),
                                       require_frequency=True)

    def test_unlabelled_manifest_requires_paths(self):
        observed = prepare_manifest(pd.DataFrame({"sample": ["new"], "file_path": ["new.csv"]}))
        self.assertNotIn("true_label", observed)
        with self.assertRaises(InputFormatError):
            prepare_manifest(pd.DataFrame({"sample": ["new"]}))

    def test_internal_external_add_and_clinical_names(self):
        self.assertEqual(evaluation_set_from_name("RA_internal_add.csv"), "internal")
        self.assertEqual(evaluation_set_from_name("T1D_external_add.csv"), "external")
        self.assertIsNone(evaluation_set_from_name("CeD_clinical .csv"))

    def test_missing_original_padding_settings_are_rejected(self):
        settings = InferenceSettings.from_mapping({
            "checkpoint": "a", "bert_config": "b", "vocab_file": "c"})
        with self.assertRaises(ConfigurationError):
            settings.model_options()

    def test_pooler_is_explicitly_rejected(self):
        settings = InferenceSettings.from_mapping({
            "checkpoint": "a", "bert_config": "b", "vocab_file": "c",
            "max_length": 32, "dropout": 0.2, "pooling_type": "pooler"})
        with self.assertRaises(ConfigurationError):
            settings.model_options()

    def test_logits_are_not_accepted_as_probabilities(self):
        class BadBackend:
            def predict_proba(self, sequences):
                return np.array([[4.0, -2.0]])
        from autotcr.exceptions import ModelLoadError
        with self.assertRaises(ModelLoadError):
            AutoTCRPredictor(BadBackend()).predict_sequences(pd.DataFrame({"sequence": ["ABC"]}))

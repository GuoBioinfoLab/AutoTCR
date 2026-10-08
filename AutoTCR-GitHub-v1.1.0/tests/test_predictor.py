import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autotcr.predictor import AutoTCRPredictor


class DummyBackend:
    def predict_proba(self, sequences):
        positive = np.asarray([0.8 if "HIGH" in value else 0.2 for value in sequences])
        return np.column_stack([1.0 - positive, positive])


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.predictor = AutoTCRPredictor(DummyBackend())

    def test_sequence_order_is_preserved(self):
        table = pd.DataFrame({"sequence": ["LOW", "HIGH", "LOW"]})
        result = self.predictor.predict_sequences(table)
        np.testing.assert_allclose(result["autoimmune_probability"], [0.2, 0.8, 0.2])
        self.assertEqual(result["input_row"].tolist(), [0, 1, 2])

    def test_repertoire_contributions_sum_to_ars(self):
        table = pd.DataFrame({"sequence": ["LOW", "HIGH"], "Freq": [1, 3]})
        result = self.predictor.predict_repertoire(table, sample_id="sample-1")
        self.assertAlmostEqual(result.summary["ars"], 0.65)
        self.assertAlmostEqual(result.per_sequence["ars_contribution"].sum(), 0.65)

    def test_cohort_path_resolution_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disease = root / "disease"
            healthy = root / "healthy"
            disease.mkdir()
            healthy.mkdir()
            pd.DataFrame({"TcRb_vj": ["HIGH"], "Freq": [2]}).to_csv(
                disease / "case_1_input_seq_vj.csv", index=False
            )
            pd.DataFrame({"TcRb_vj": ["LOW"], "Freq": [2]}).to_csv(
                healthy / "control_1_input_seq_vj.csv", index=False
            )
            manifest = pd.DataFrame(
                {
                    "sample": ["case_1", "control_1"],
                    "true_label": [1, 0],
                    "disease": ["AIH", "Healthy"],
                }
            )
            result = self.predictor.predict_cohort(
                manifest,
                disease_dir=disease,
                healthy_dir=healthy,
            )
            self.assertEqual(result["status"].tolist(), ["ok", "ok"])
            np.testing.assert_allclose(result["ars"], [0.8, 0.2])


if __name__ == "__main__":
    unittest.main()


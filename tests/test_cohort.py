import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotcr.cohort import discover_manifests


class CohortDiscoveryTests(unittest.TestCase):
    def test_internal_external_are_loaded_and_clinical_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = pd.DataFrame({"sample": ["s1"], "true_label": [1]})
            base.to_csv(root / "AIH_internal.csv", index=False)
            base.to_csv(root / "AIH_external.csv", index=False)
            base.to_csv(root / "AIH_clinical.csv", index=False)
            result = discover_manifests(root)
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result["evaluation_set"]), {"internal", "external"})


if __name__ == "__main__":
    unittest.main()


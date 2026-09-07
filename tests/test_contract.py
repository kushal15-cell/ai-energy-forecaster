"""Regression checks for leakage, holdout isolation, and real inference behavior."""

import unittest

import numpy as np
import pandas as pd

from src.final_feature_engineering import FEATURES, ROOT, prepare
from src.inference import DEFAULT, artifacts, forecast, make_features, scenarios


class ContractTests(unittest.TestCase):
    def test_no_outcome_or_provenance_predictors(self):
        forbidden = {
            "duration_seconds",
            "log_duration",
            "source",
            "label_quality",
            "co2_kg",
            "energy_kwh",
        }
        model, meta = artifacts()
        self.assertFalse(set(FEATURES) & forbidden)
        self.assertEqual(list(model.feature_names_in_), FEATURES)
        self.assertEqual(meta["features"], FEATURES)

    def test_external_isolation_and_complete_split(self):
        split = pd.read_csv(ROOT / "reports/split_manifest.csv")
        self.assertFalse(split.row_id.duplicated().any())
        self.assertTrue(
            (split[split.split != "external_test"].source == "synthetic").all()
        )
        self.assertTrue(
            (split[split.source != "synthetic"].split == "external_test").all()
        )
        data = pd.read_csv(ROOT / "data/final_training_dataset.csv")
        self.assertEqual(set(data.row_id), set(split.row_id))
        train = data[data.row_id.isin(split[split.split == "train"].row_id)]
        np.testing.assert_allclose(
            artifacts()[0]["preprocess"].named_transformers_["numeric"].statistics_,
            train[FEATURES[:6]].median().values,
        )

    def test_duration_changes_cannot_change_features(self):
        data = pd.read_csv(ROOT / "data/merged_dataset.csv").head(10)
        changed = data.assign(duration_seconds=1e12, co2_kg=1e9)
        pd.testing.assert_frame_equal(
            prepare(data)[FEATURES], prepare(changed)[FEATURES]
        )

    def test_preserves_existing_hf_compute_estimate(self):
        raw = pd.read_csv(ROOT / "data/merged_dataset.csv")
        hf = raw[raw.source == "huggingface"].head(1)
        expected = np.log10(10.0 ** hf.log_flops.iloc[0] + 1)
        self.assertAlmostEqual(prepare(hf).log_flops.iloc[0], expected)

    def test_shap_additivity_and_valid_interval(self):
        result = forecast(DEFAULT)
        reconstructed = result["base_log10"] + sum(result["contributions"].values())
        self.assertAlmostEqual(reconstructed, np.log10(result["energy_kwh"]), places=4)
        self.assertLess(result["lower_kwh"], result["energy_kwh"])
        self.assertGreater(result["upper_kwh"], result["energy_kwh"])
        self.assertFalse(result["warnings"])

    def test_scenarios_recompute_coupled_features(self):
        variants = scenarios(DEFAULT)
        changed = {**DEFAULT, "estimated_tokens": DEFAULT["estimated_tokens"] * 0.8}
        self.assertAlmostEqual(
            variants.iloc[1].energy_kwh, forecast(changed, False)["energy_kwh"]
        )
        self.assertLess(
            make_features(changed).log_flops.iloc[0],
            make_features(DEFAULT).log_flops.iloc[0],
        )

    def test_unseen_and_invalid_inputs(self):
        self.assertTrue(
            forecast({**DEFAULT, "task": "never-seen", "n_params": 1e12}, False)[
                "warnings"
            ]
        )
        with self.assertRaises(ValueError):
            make_features({**DEFAULT, "n_params": -1})


if __name__ == "__main__":
    unittest.main()

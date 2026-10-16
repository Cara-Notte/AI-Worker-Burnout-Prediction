"""Regression checks for leakage, data retention, and inference edge cases."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("burnout", Path(__file__).resolve().parents[1] / "code.py")
burnout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burnout)


class PipelineTests(unittest.TestCase):
    def test_training_statistics_and_unseen_categories(self):
        train = pd.DataFrame({"work_hours": [1., 2., np.nan, 4.], "job_role": ["A", "B", np.nan, "A"]})
        pipeline = burnout.make_models(train)["Linear Regression"]
        pipeline.fit(train, pd.Series([10., 20., 25., 40.]))
        numeric = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        mean_before = numeric.named_steps["scaler"].mean_.copy()
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
        categories_before = encoder.categories_[0].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prediction = pipeline.predict(pd.DataFrame({"work_hours": [1000000., np.nan], "job_role": ["NEW", np.nan]}))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(mean_before, numeric.named_steps["scaler"].mean_)
        self.assertAlmostEqual(mean_before[0], 2.25)
        np.testing.assert_array_equal(categories_before, encoder.categories_[0])
        self.assertNotIn("NEW", encoder.categories_[0])

    def test_extremes_retained_and_outcome_excluded(self):
        frame = pd.DataFrame({"employee_id": ["A", "B", "C"], "burnout_score": [10, 50, 1000],
                              "attrition_risk": ["Low", "Medium", "High"], "hours": [1, 2, 9999]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            frame.to_csv(path, index=False)
            cleaned, audit = burnout.load_data(path)
        self.assertEqual(audit["retained_rows"], 3)
        X, y = burnout.make_features(cleaned)
        self.assertNotIn("attrition_risk", X)
        self.assertNotIn("burnout_score", X)
        self.assertEqual(y.max(), 1000)

    def test_conflicting_identity_rejected(self):
        frame = pd.DataFrame({"employee_id": ["A", "A"], "burnout_score": [10, 50],
                              "attrition_risk": ["Low", "Medium"]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "employee IDs"):
                burnout.load_data(path)


if __name__ == "__main__":
    unittest.main()

"""Regression checks for the frozen TOPPQuad export and baseline geometry."""
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial import Polynomial

from geometric_validation.reference import build_reference
from geometric_validation.runtime import digest, load_config


class ToppQuadExportTest(unittest.TestCase):
    def test_frozen_export_preserves_each_baseline_path(self):
        root = Path(__file__).resolve().parents[1]
        baseline_path = root / "configs/geometric_planners_v1.json"
        baseline = load_config(baseline_path)
        retimed = load_config(root / "configs/geometric_toppquad_v1.json")
        self.assertEqual(retimed["comparison"]["baseline_config_sha256"], digest(baseline_path))
        self.assertEqual(retimed["cases"].keys(), baseline["cases"].keys())
        for key in ("airsim", "controller", "execution", "limits", "visualization"):
            self.assertEqual(retimed[key], baseline[key])

        for case in baseline["cases"]:
            with self.subTest(case=case):
                before = build_reference(baseline, case)
                after = build_reference(retimed, case)
                planning = retimed["cases"][case]["frozen_reference"]["planning"]
                warp = Polynomial(planning["progress_polynomial"])
                for fraction in np.linspace(0., 1., 201):
                    actual = after.sample(fraction * after.duration)
                    expected = before.sample(np.clip(warp(fraction), 0., 1.) * before.duration)
                    np.testing.assert_allclose(actual["p"], expected["p"], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(
                        actual["v"], np.asarray(expected["v"]) * warp.deriv()(fraction)
                        * before.duration / after.duration, atol=1e-8, rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()

"""Configuration integration checks; no simulator or optimizer is started."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .core import BezierParameters
from .runtime import ControlRuntime
from .trajectory_profile import load_trajectory_profile


class TrajectoryProfileTests(unittest.TestCase):
    def test_missing_profile_is_exact_baseline_and_invalid_profile_fails(self):
        self.assertEqual(load_trajectory_profile({})[0], BezierParameters())
        for section in (None, [], {"parameters": {"outgoing_weigth": .4}},
                        {"parameters": {"pass_speed_scale": True}}, {"name": ""}):
            with self.subTest(section=section), self.assertRaises(ValueError):
                load_trajectory_profile({"trajectory": section})

    def test_profile_is_applied_and_saved_without_online_search(self):
        config = {"control": {"seed": 13}, "trajectory": {
            "name": "offline candidate", "parameters": {"outgoing_weight": .2}}}
        runtime = ControlRuntime(config, mode="demo")
        try:
            self.assertEqual(runtime.controller.parameters.outgoing_weight, .2)
            self.assertEqual(runtime.snapshot()["trajectory_name"], "offline candidate")
            self.assertFalse(runtime._active)
            with tempfile.TemporaryDirectory() as folder:
                runtime._run_dir = Path(folder)
                runtime._save("TEST_NO_FLIGHT")
                report = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(report["bezier_parameters"]["outgoing_weight"], .2)
                self.assertEqual(report["trajectory_name"], "offline candidate")
                runtime._run_dir = None
            controller = runtime.controller
            with self.assertRaises(ValueError):
                runtime._handle("configure", {"trajectory": {"parameters": {"outgoing_weight": 8}}})
            self.assertIs(runtime.controller, controller)
            self.assertEqual(runtime.config, config)
            runtime._handle("configure", {"control": {"seed": 14}})
            self.assertEqual(runtime.controller.parameters, BezierParameters())
            self.assertEqual(runtime.snapshot()["bezier_parameters"], BezierParameters().to_dict())
        finally:
            runtime.shutdown()
            self.assertTrue(runtime.wait_closed(2))

    def test_runtime_import_does_not_import_torch_or_optimizer(self):
        python_root = Path(__file__).resolve().parents[1]
        code = ("import sys; import dual_target_control.runtime; "
                "assert 'torch' not in sys.modules; "
                "assert 'dual_target_control.adaptive_search' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=python_root,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

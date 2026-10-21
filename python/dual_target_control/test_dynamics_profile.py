"""Source/mirror and independent physics checks; this does not execute C++ or fly.

The UE build and the fresh DTC_DYNAMICS receipt remain the runtime checks.
Run from the repository root: python -m unittest discover -s python/dual_target_control -p test_dynamics_profile.py
"""

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
AIRSIM = ROOT / "external" / "AirSim"
AIRLIB = AIRSIM / "AirLib" / "include"
MODEL = Path("vehicles/multirotor/firmwares/simple_flight/SimpleFlightQuadXParams.hpp")
PARAMS = Path("vehicles/multirotor/MultiRotorParams.hpp")
ROTOR = Path("vehicles/multirotor/RotorParams.hpp")
FIRMWARE = Path("vehicles/multirotor/firmwares/simple_flight/firmware/Params.hpp")
PLUGIN_SOURCES = (
    AIRSIM / "Unreal/Plugins/AirSim/Source",
    AIRSIM / "Unreal/Environments/Blocks/Plugins/AirSim/Source",
)


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DynamicsProfileSourceTests(unittest.TestCase):
    def test_all_model_and_receipt_mirrors_match(self):
        for relative in (MODEL, PARAMS, ROTOR, FIRMWARE):
            baseline = (AIRLIB / relative).read_bytes()
            for plugin in PLUGIN_SOURCES:
                with self.subTest(file=relative, plugin=plugin):
                    self.assertEqual(baseline, (plugin / "AirLib/include" / relative).read_bytes())
        relative = Path("Vehicles/Multirotor/MultirotorPawnSimApi.cpp")
        self.assertEqual(*[(plugin / relative).read_bytes() for plugin in PLUGIN_SOURCES])

    def test_legacy_generic_quad_and_controller_are_unchanged(self):
        params = (AIRLIB / PARAMS).read_text(encoding="utf-8")
        quad = re.search(
            r"        void setupFrameGenericQuad\(Params& params\).*?(?=        void setupFrameGenericHex)",
            params, re.S,
        ).group()
        # These are the unmodified source hashes from before this opt-in change.
        self.assertEqual(text_digest(quad), "6fff5d129d32d603fc518c0996e85d72b2361737cc7ce95128fbde3eb714f432")
        self.assertEqual(text_digest((AIRLIB / ROTOR).read_text(encoding="utf-8")),
                         "c8f6288a6a5e5a45a28acb4e9222d2e572c2f2ac1c1e181685c0b9caa2a9addf")
        self.assertEqual(text_digest((AIRLIB / FIRMWARE).read_text(encoding="utf-8")),
                         "cb94d03e4ed898b8251eea6b2060f8604b17ecee37b250e020c07935bbb66b95")

    def test_opt_in_precedes_override_and_recomputes_rotor_outputs(self):
        model = (AIRLIB / MODEL).read_text(encoding="utf-8")
        self.assertLess(model.index("setupFrameGenericQuad(params);"), model.index("applyDynamicsProfile(params);"))
        no_profile = model.index('!vehicle.hasKey("DynamicsProfile")')
        self.assertLess(no_profile, model.index("params.mass = static_cast<real_T>(mass_kg)"))
        self.assertNotRegex(model, r"max_(?:thrust|torque)\s*=(?!=)")
        self.assertLess(model.index("params.rotor_params.max_rpm ="),
                        model.index("params.rotor_params.calculateMaxThrust();"))
        self.assertIn("computeInertiaMatrix(params.inertia", model)
        self.assertIn("!std::isfinite(mass_kg)", model)
        self.assertIn("!std::isfinite(twr)", model)

    def test_receipt_uses_ue_log_after_physics_body_creation(self):
        source = (PLUGIN_SOURCES[1] / "Vehicles/Multirotor/MultirotorPawnSimApi.cpp").read_text(encoding="utf-8")
        receipt = source.index('UE_LOG(LogTemp, Display, TEXT("DTC_DYNAMICS %s")')
        self.assertLess(source.index("new MultiRotorPhysicsBody"), receipt)
        self.assertIn("if (!dynamics_receipt.empty())", source[:receipt])

    def test_task_profile_has_explicit_schema_and_operating_margin(self):
        settings = json.loads((ROOT / "configs/dual_target_airsim_settings.json").read_text(encoding="utf-8-sig"))
        profile = settings["Vehicles"]["DroneControl"]["DynamicsProfile"]
        self.assertEqual(profile, {
            "Schema": 1, "Name": "dual_target_v1", "MassKg": 1.0, "ThrustToWeightRatio": 2.5,
        })
        hover_throttle = 1 / profile["ThrustToWeightRatio"]
        # VelocityController clamps to 0.3. At 2.5 the vehicle can still descend.
        self.assertAlmostEqual(hover_throttle, 0.4)
        self.assertLess(0.3, hover_throttle)
        self.assertLess(hover_throttle, 1.0)

    def test_independent_propeller_force_and_torque_reference(self):
        rotor_source = (AIRLIB / ROTOR).read_text(encoding="utf-8")
        values = {}
        for key in ("C_T", "C_P", "air_density", "max_rpm", "propeller_diameter"):
            values[key] = float(re.search(r"real_T " + key + r" = ([0-9.]+)f", rotor_source).group(1))
        d = values["propeller_diameter"]
        original_thrust = values["C_T"] * values["air_density"] * (values["max_rpm"] / 60) ** 2 * d ** 4
        thrust_per_rotor = 1.0 * 9.80665 * 2.5 / 4
        # UIUC nondimensional coefficients imply this torque/thrust ratio,
        # independently of mass, density and motor speed.
        torque_per_rotor = thrust_per_rotor * values["C_P"] / values["C_T"] * d / (2 * math.pi)
        required_rpm = 60 * math.sqrt(thrust_per_rotor / (values["C_T"] * values["air_density"] * d ** 4))
        self.assertAlmostEqual(4 * original_thrust / 9.80665, 1.7047396481104402)
        self.assertAlmostEqual(thrust_per_rotor, 6.12915625)
        self.assertAlmostEqual(torque_per_rotor, 0.081481917445498)
        self.assertAlmostEqual(required_rpm, 7746.307199254776)
        self.assertGreater(required_rpm, values["max_rpm"])


if __name__ == "__main__":
    unittest.main()

"""Freeze offline V4 matched-start lateral pilot configurations.

This prepares inputs and a paired session plan. It never connects to AirSim,
starts a flight, or records a planned session as an executed result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs/single_trajectory_fixed_height_v3_extended.json"
DEFAULT_OUTPUT = ROOT / "configs/trajectory_matched_pilot_v4"
DEFAULT_SETTINGS = ROOT / "configs/settings_trajectory_interface.json"
PROTOCOL_ID = "single_trajectory_matched_start_lateral_v4_20261003"
RESEARCH_SCOPE = "development_matched_start_paired_lateral_pilot_fixed_height_preview"
CONDITIONS = (("center", 0.0), ("target_y_minus_0p4", -0.4), ("target_y_plus_0p4", 0.4))
INITIAL_STATE_MATCH = {
    "enabled": True,
    "position_tolerance_m": 0.015,
    "speed_tolerance_m_s": 0.01,
    "acceleration_tolerance_m_s2": 0.05,
    "attitude_tolerance_deg": 1.0,
    "stable_s": 0.5,
    "timeout_s": 30.0,
    "reposition_speed_cap_m_s": 0.05,
    "reposition_acceleration_cap_m_s2": 0.05,
    "proportional_gain": 0.8,
    "command_dt_s": 0.1,
    "max_sample_gap_s": 0.25,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, value: dict) -> None:
    # Exclusive creation protects both an existing freeze and a race with
    # another writer. A partial preparation is retained if a later write fails.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _require_numeric(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _require_vector(value, expected, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must have three coordinates")
    actual = [_require_numeric(part, name) for part in value]
    if actual != list(expected):
        raise ValueError(f"{name} differs from the inspected development baseline")


def validate_base(base: dict) -> None:
    """Reject changed geometry/budgets before creating the output directory."""
    for name, expected in (
        ("target_world_ned_m", (0.0, 0.0, -3.0)),
        ("observation_world_ned_m", (-5.0, -0.5, -3.0)),
        ("workspace_min_ned_m", (-5.6, -1.3, -3.6)),
        ("workspace_max_ned_m", (-2.4, 0.6, -2.4)),
    ):
        _require_vector(base[name], expected, name)
    if base["vehicle"] != "DroneA" or base["target"] != "BalloonTarget":
        raise ValueError("Only the inspected single DroneA / BalloonTarget baseline is supported")
    for group, name, expected in (
        ("safety", "coordinate_speed_m_s", 0.15),
        ("safety", "approach_speed_cap_m_s", 0.15),
        ("visual", "frame_max_age_s", 0.5),
        ("planning", "execution_speed_cap_m_s", 0.15),
        ("planning", "max_acceleration_m_s2", 0.3),
        ("planning", "fixed_world_height_ned_m", -3.0),
    ):
        if _require_numeric(base[group][name], f"{group}.{name}") != expected:
            raise ValueError(f"{group}.{name} differs from the inspected development budget")
    if base["planning"]["execution_mode"] != "fixed_world_height_preview":
        raise ValueError("The V4 pilot retains fixed-height preview execution")
    if base["planning"]["candidate_steps_z_m"] != [0.0]:
        raise ValueError("The V4 pilot requires the existing zero vertical candidate step")


def build_configs(base: dict, seed: int) -> dict[str, dict]:
    validate_base(base)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Order seed must be a nonnegative integer")
    common = deepcopy(base)
    common.update(protocol_id=PROTOCOL_ID, research_stage=RESEARCH_SCOPE,
                  formal_thesis_effect_results=False,
                  initial_state_match=deepcopy(INITIAL_STATE_MATCH))
    common["evidence_limits"] = list(common.get("evidence_limits", [])) + [
        "Matched-start development pilot with measured position/velocity/acceleration/attitude acceptance",
        "Two repetitions per condition in independent Unreal sessions; not formal thesis effectiveness data",
        "Order seed controls planned condition ordering only; not an Unreal scene randomization seed",
    ]
    configs = {}
    for condition_id, offset_y in CONDITIONS:
        config = deepcopy(common)
        config["target_world_ned_m"][1] = offset_y
        config.update(condition_id=condition_id, target_offset_ned_m=[0.0, offset_y, 0.0],
                      condition_metadata={
                          "condition_id": condition_id,
                          "world_ned_y_offset_m": offset_y,
                          "planned_repetitions": [1, 2],
                          "order_seed": seed,
                          "method_orders_by_repetition": {
                              "1": ["efficiency", "perception"],
                              "2": ["perception", "efficiency"],
                          },
                          "order_seed_scope": "lateral_condition_order_only_not_scene_seed",
                      })
        configs[condition_id] = config
    return configs


def session_plan(seed: int, config_paths: dict[str, Path]) -> list[dict]:
    lateral = [name for name, _ in CONDITIONS if name != "center"]
    random.Random(seed).shuffle(lateral)
    first_order = ["center"] + lateral
    sessions = []
    for repetition, order in ((1, first_order), (2, list(reversed(first_order)))):
        for condition_id in order:
            reverse = repetition == 2
            sessions.append({
                "planned_session_index": len(sessions) + 1,
                "condition_id": condition_id,
                "repetition": repetition,
                "independent_unreal_session": True,
                "configuration_path": str(config_paths[condition_id]),
                "target_source": "sensor_surface_estimate",
                "method_order": ["perception", "efficiency"] if reverse else ["efficiency", "perception"],
                "reverse_order": reverse,
                "order_seed": seed,
                "execution_status": "NOT_RUN",
            })
    return sessions


def prepare_conditions(base_path: Path, output_dir: Path, seed: int = 20261003,
                       settings_path: Path = DEFAULT_SETTINGS) -> Path:
    base_path, output_dir, settings_path = (Path(path).resolve() for path in
                                            (base_path, output_dir, settings_path))
    # Check all inputs before making a directory; the generator never changes
    # the V3 source configuration or an already frozen V4 directory.
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))
    configs = build_configs(base, seed)
    source_inputs = {
        "base_config": {"path": str(base_path), "sha256": sha256(base_path)},
        "settings": {"path": str(settings_path), "sha256": sha256(settings_path)},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    config_paths = {}
    rows = []
    for condition_id, config in configs.items():
        path = output_dir / f"{condition_id}.json"
        _write_new_json(path, config)
        config_paths[condition_id] = path
        rows.append({"condition_id": condition_id, "configuration_path": str(path),
                     "configuration_sha256": sha256(path),
                     "target_world_ned_m": config["target_world_ned_m"],
                     "target_offset_ned_m": config["target_offset_ned_m"],
                     "planned_repetitions": 2})
    plan = session_plan(seed, config_paths)
    manifest = {
        "schema_version": 1,
        "kind": "MATCHED_START_LATERAL_PILOT_CONFIG_FREEZE",
        "protocol_id": PROTOCOL_ID,
        "research_scope": RESEARCH_SCOPE,
        "preparation_status": "PREPARED_NOT_EXECUTED",
        "formal_thesis_effect_results": False,
        "prepared_local": datetime.now().astimezone().isoformat(),
        "order_seed": seed,
        "order_seed_scope": "lateral_condition_order_only_not_scene_seed",
        "source_inputs": source_inputs,
        "initial_state_match": deepcopy(INITIAL_STATE_MATCH),
        "planned_condition_count": len(configs),
        "planned_independent_unreal_sessions": len(plan),
        "planned_flights_per_session": 2,
        "planned_flights": len(plan) * 2,
        "method_configuration_scope": "Same condition configuration shared by both methods",
        "planned_primary_target_source": "sensor_surface_estimate",
        "gate_scope": "Root operator checks the short matched-start gate before progressive execution",
        "conditions": rows,
        "session_plan": plan,
        "execution_results": [],
        "limitations": [
            "Prepared configuration and order only; all session entries are NOT_RUN",
            "Fixed-height positive-time preview, not complete three-dimensional trajectory execution",
            "No method superiority claim or formal experiment result is produced",
            "A seeded condition order does not randomize Unreal dynamics or establish independent errors",
        ],
        "ai_assistance": "Configuration generator and offline tests authored with AI assistance",
    }
    manifest_path = output_dir / "conditions.json"
    _write_new_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    print(prepare_conditions(args.base_config, args.output_dir, args.seed, args.settings))


if __name__ == "__main__":
    main()

"""Materialize prespecified small lateral offsets with an unchanged waypoint baseline."""
import copy
import json
from pathlib import Path

from single_drone_experiment import load_config
from setup_scene import ROOT, validate_target_offset

OFFSETS = {"center": [0.0, 0.0, 0.0],
           "target_y_minus_06": [0.0, -0.6, 0.0],
           "target_y_plus_06": [0.0, 0.6, 0.0]}


def build(base, origin):
    conditions = {}
    for name, offset in OFFSETS.items():
        validate_target_offset(offset)
        cfg = copy.deepcopy(base)
        cfg["condition_id"] = name
        cfg["protocol_id"] = "phase1_lateral_pilot_20260930"
        cfg["target_world_ned_m"] = [float(x + y) for x, y in zip(origin, offset)]
        conditions[name] = cfg
    return conditions


def main():
    base = load_config(ROOT / "configs/single_drone_matched_speed.json")
    origin = json.loads((ROOT / "configs/balloon.json").read_text())["position_ned_m"]
    folder = ROOT / "configs/phase1_lateral_pilot"
    folder.mkdir(exist_ok=True)
    for name, cfg in build(base, origin).items():
        path = folder / f"{name}.json"
        text = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Preserve previously materialized condition: {path}")
        path.write_text(text, encoding="utf-8")
        load_config(path)
        print(path)


if __name__ == "__main__":
    main()

"""Repeatable turning checks using a lagged point mass, never AirSim physics.

The plant deliberately lags both velocity and measured yaw. It is useful for
regressions that a command-following demo yaw would hide; it does not model
rotors, attitude, thrust limits, or validate performance of a real aircraft.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/dual_target_control/turn_fix_20261014"


@dataclass(frozen=True)
class TurningScenario:
    name: str
    position: tuple
    velocity: tuple
    targets: tuple
    config: dict
    yaw: float = 0.0
    deadline: float = 120.0


SCENARIOS = (
    TurningScenario("right_angle", (0, 0, 5), (0, 0, 0),
                    ((10, 0, 5), (10, 10, 5)), {}),
    TurningScenario("near_reversal", (0, 0, 5), (0, 0, 0),
                    ((10, 0, 5), (0, 1, 5)), {}),
    TurningScenario("near_boundary", (10, 0, 5), (3, 0, 0),
                    ((18, 0, 5), (10, 8, 5)), {}),
    TurningScenario("ascend_then_descend", (0, 0, 4), (0, 0, 0),
                    ((10, 0, 8.5), (10, 10, 3.5)), {}),
    TurningScenario("descend_then_ascend", (0, 0, 8.5), (0, 0, 0),
                    ((10, 0, 3.5), (10, 10, 8.5)), {}),
    TurningScenario(
        "recorded_long_approach", (0, 0, 7.82720947265625),
        (0, 0, 0.24865186214447021),
        ((35.750163527625475, 34.81964646334178, 20.031491374471184),
         (-23.378256581185294, -41.4344531435947, 22.87578138936733)),
        dict(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0,
             height_min=2.0, height_max=25.0)),
)
INTERVALS = ((0.02,), (0.02, 0.02, 0.04, 0.06))
YAW_LAGS = (0.25, 0.35)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sphere_entry(previous, position, target, radius):
    """Return the first entry fraction, including an already-inside segment.

    This is independent of controller internals and catches swept crossings
    whose two sampled endpoints both lie outside the arrival sphere.
    """
    offset = previous - target
    c = float(np.dot(offset, offset) - radius * radius)
    if c <= 1e-12:
        return 0.0
    step = position - previous
    a = float(np.dot(step, step))
    if a <= 1e-18:
        return None
    b = 2 * float(np.dot(offset, step))
    discriminant = b * b - 4 * a * c
    if discriminant < -1e-12:
        return None
    fraction = (-b - math.sqrt(max(0.0, discriminant))) / (2 * a)
    return float(np.clip(fraction, 0.0, 1.0)) if -1e-10 <= fraction <= 1 + 1e-10 else None


def entry_yaw_error(previous, position, target, previous_yaw, yaw, fraction, fallback):
    entry_point = previous + fraction * (position - previous)
    delta = target - entry_point
    reference = math.atan2(float(delta[1]), float(delta[0])) if np.linalg.norm(delta[:2]) >= 1e-7 else fallback
    measured = wrap(previous_yaw + fraction * wrap(yaw - previous_yaw))
    return abs(math.degrees(wrap(reference - measured)))


def _new_leg(target, position, yaw):
    delta = target - position
    inbound = delta / max(float(np.linalg.norm(delta)), 1e-12)
    return dict(target=target.tolist(), first_entry_time_s=None,
                first_entry_yaw_error_deg=None, first_entry_granted=False,
                first_hit_time_s=None, hit_yaw_error_deg=None,
                hit_speed_mps=None, minimum_distance_m=float("inf"),
                near_reversal_count=0, plane_crossings_before_hit=0,
                geometric_entry_count=0, _inbound=inbound,
                _near_min=float("inf"), _reversal_latched=False,
                _passed_plane=False, _inside=False, _fallback=yaw)


def run_scenario(core, scenario, *, velocity_lag=0.30, yaw_lag=0.25,
                 intervals=(0.02,), include_trace=False):
    # Keep the comparison's arrival requirements and limits fixed even if a
    # future production default changes. The cure must not be a wider sphere.
    options = dict(arrival_radius=0.5, heading_tolerance_deg=20.0,
                   max_speed=4.0, max_acceleration=4.0,
                   max_yaw_rate_deg=120.0, seed=13)
    options.update(scenario.config)
    config = core.ControlConfig(**options).validate()
    controller = core.RollingController(config)
    position = np.array(scenario.position, dtype=float)
    velocity = np.array(scenario.velocity, dtype=float)
    yaw = float(scenario.yaw)
    controller.reset(position, velocity, yaw)
    controller.set_targets(scenario.targets)
    original_targets = [np.array(target, dtype=float) for target in scenario.targets]
    legs = [_new_leg(original_targets[0], position, yaw)]
    initial = controller.snapshot()
    command = initial["velocity_command"].copy()
    yaw_command = math.radians(initial["yaw_command_deg"])
    previous_acceleration = previous_yaw_rate = None
    maxima = dict(command_speed_mps=0.0, command_acceleration_mps2=0.0,
                  command_yaw_rate_deg_s=0.0, command_jerk_mps3=0.0,
                  command_yaw_acceleration_deg_s2=0.0)
    violations = dict(speed=0, acceleration=0, yaw_rate=0, bounds=0,
                      planned_bounds=0, nonfinite=0, arrival_gate=0)
    queue_updated = False
    target_order = []
    trace = []
    elapsed, step_number = 0.0, 0
    while elapsed < scenario.deadline and len(target_order) < 2:
        dt = float(intervals[step_number % len(intervals)])
        previous, previous_yaw, previous_time = position.copy(), yaw, elapsed
        previous_command, previous_yaw_command = command.copy(), yaw_command
        # Zero-order-held commands enter a first-order lagged point-mass plant.
        velocity += (command - velocity) * (1 - math.exp(-dt / velocity_lag))
        position += velocity * dt
        yaw = wrap(yaw + wrap(yaw_command - yaw) * (1 - math.exp(-dt / yaw_lag)))
        elapsed += dt
        state = controller.update(position, velocity, yaw, dt)
        command = state["velocity_command"].copy()
        yaw_command = math.radians(state["yaw_command_deg"])
        speed = float(np.linalg.norm(command))
        acceleration = (command - previous_command) / dt
        yaw_rate = math.degrees(wrap(yaw_command - previous_yaw_command)) / dt
        maxima["command_speed_mps"] = max(maxima["command_speed_mps"], speed)
        maxima["command_acceleration_mps2"] = max(maxima["command_acceleration_mps2"], float(np.linalg.norm(acceleration)))
        maxima["command_yaw_rate_deg_s"] = max(maxima["command_yaw_rate_deg_s"], abs(yaw_rate))
        if previous_acceleration is not None:
            maxima["command_jerk_mps3"] = max(maxima["command_jerk_mps3"], float(np.linalg.norm(acceleration - previous_acceleration)) / dt)
            maxima["command_yaw_acceleration_deg_s2"] = max(maxima["command_yaw_acceleration_deg_s2"], abs(yaw_rate - previous_yaw_rate) / dt)
        previous_acceleration, previous_yaw_rate = acceleration, yaw_rate
        violations["speed"] += int(speed > config.max_speed + 1e-8)
        violations["acceleration"] += int(np.linalg.norm(acceleration) > config.max_acceleration + 1e-8)
        violations["yaw_rate"] += int(abs(yaw_rate) > config.max_yaw_rate_deg + 1e-8)
        violations["bounds"] += int(np.any(position < config.lower - 1e-6) or np.any(position > config.upper + 1e-6))
        path = state["planned_path"]
        violations["planned_bounds"] += int(np.any(path < config.inner_lower - 1e-8) or np.any(path > config.inner_upper + 1e-8))
        violations["nonfinite"] += int(not (np.isfinite(position).all() and np.isfinite(velocity).all()
                                            and np.isfinite(command).all() and math.isfinite(yaw_command)))
        leg = legs[-1]
        target = original_targets[len(target_order)]
        distance = float(np.linalg.norm(position - target))
        leg["minimum_distance_m"] = min(leg["minimum_distance_m"], distance)
        hit = state["hit_event"]
        fraction = sphere_entry(previous, position, target, config.arrival_radius)
        if fraction is not None and not leg["_inside"]:
            leg["geometric_entry_count"] += 1
            if leg["first_entry_time_s"] is None:
                leg["first_entry_time_s"] = previous_time + fraction * dt
                leg["first_entry_yaw_error_deg"] = entry_yaw_error(
                    previous, position, target, previous_yaw, yaw, fraction, leg["_fallback"])
                leg["first_entry_granted"] = hit is not None
        leg["_inside"] = distance <= config.arrival_radius
        if hit is None:
            if distance < 2.0:
                leg["_near_min"] = min(leg["_near_min"], distance)
            if math.isfinite(leg["_near_min"]):
                if distance > leg["_near_min"] + 0.25 and not leg["_reversal_latched"]:
                    leg["near_reversal_count"] += 1
                    leg["_reversal_latched"] = True
                elif distance <= leg["_near_min"] + 0.05:
                    leg["_reversal_latched"] = False
            passed = float(np.dot(position - target, leg["_inbound"])) >= 0.0
            if passed and not leg["_passed_plane"]:
                leg["plane_crossings_before_hit"] += 1
            leg["_passed_plane"] = passed
        else:
            target_order.append(bool(np.allclose(hit["target"], target, atol=1e-9)))
            leg["first_hit_time_s"] = elapsed
            leg["hit_yaw_error_deg"] = float(hit["yaw_error_deg"])
            leg["hit_speed_mps"] = float(hit["speed"])
            independent_error = entry_yaw_error(previous, position, target, previous_yaw, yaw,
                                                fraction if fraction is not None else 0.0, leg["_fallback"])
            violations["arrival_gate"] += int(fraction is None or independent_error > config.heading_tolerance_deg + 1e-7)
            if len(target_order) == 1:
                queue_updated = bool(np.allclose(state["targets"][0], original_targets[1], atol=1e-9)
                                     and len(state["targets"]) == 2)
                legs.append(_new_leg(original_targets[1], position, yaw))
        if include_trace:
            trace.append(dict(t=elapsed, position=position.tolist(), velocity=velocity.tolist(),
                              yaw_deg=math.degrees(yaw), command=command.tolist(),
                              yaw_command_deg=math.degrees(yaw_command), hits=int(state["hits"])))
        step_number += 1
    public_legs = [{key: value for key, value in leg.items() if not key.startswith("_")} for leg in legs]
    for leg in public_legs:
        if not math.isfinite(leg["minimum_distance_m"]):
            leg["minimum_distance_m"] = None
    checks = dict(two_targets_reached=len(target_order) == 2 and all(target_order),
                  queue_roll_preserves_blue=queue_updated,
                  first_entries_granted=len(public_legs) == 2 and all(leg["first_entry_granted"] for leg in public_legs),
                  first_entries_face_target=len(public_legs) == 2 and all(
                      leg["first_entry_yaw_error_deg"] is not None
                      and leg["first_entry_yaw_error_deg"] <= config.heading_tolerance_deg + 1e-7 for leg in public_legs),
                  no_near_reversal=all(leg["near_reversal_count"] == 0 for leg in public_legs),
                  no_missed_plane_pass=all(leg["plane_crossings_before_hit"] == 0 for leg in public_legs),
                  command_limits=all(violations[key] == 0 for key in ("speed", "acceleration", "yaw_rate", "nonfinite")),
                  actual_and_planned_bounds=violations["bounds"] == 0 and violations["planned_bounds"] == 0,
                  accepted_hits_meet_measured_gate=violations["arrival_gate"] == 0)
    result = dict(scenario=scenario.name, velocity_lag_s=velocity_lag, yaw_lag_s=yaw_lag,
                  intervals_s=list(intervals), elapsed_s=elapsed, steps=step_number,
                  initial_position_m=list(scenario.position),
                  initial_velocity_mps=list(scenario.velocity), initial_yaw_rad=scenario.yaw,
                  deadline_s=scenario.deadline, control_config=dict(config.__dict__),
                  arrival_radius_m=config.arrival_radius, heading_tolerance_deg=config.heading_tolerance_deg,
                  targets=[target.tolist() for target in original_targets], legs=public_legs,
                  maxima=maxima, violations=violations, checks=checks,
                  passed=all(checks.values()))
    if include_trace:
        result["trace"] = trace
    return result


def load_core(path, name):
    """Execute exactly the source bytes whose digest is reported."""
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def run_matrix(core):
    return [run_scenario(core, scenario, yaw_lag=yaw_lag, intervals=intervals)
            for scenario in SCENARIOS for yaw_lag in YAW_LAGS for intervals in INTERVALS]


def summarize(rows):
    legs = [leg for row in rows for leg in row["legs"]]
    return dict(cases=len(rows), passed=sum(row["passed"] for row in rows),
                target_legs=len(legs), first_entries_granted=sum(leg["first_entry_granted"] for leg in legs),
                near_reversals=sum(leg["near_reversal_count"] for leg in legs),
                missed_plane_passes=sum(leg["plane_crossings_before_hit"] for leg in legs),
                failures=[dict(scenario=row["scenario"], yaw_lag_s=row["yaw_lag_s"],
                               intervals_s=row["intervals_s"],
                               failed_checks=[key for key, passed in row["checks"].items() if not passed])
                          for row in rows if not row["passed"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=REPORT_DIR / "baseline_core.py")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "offline_validation.json")
    args = parser.parse_args()
    current_path = ROOT / "python/dual_target_control/core.py"
    baseline, baseline_hash = load_core(args.baseline, "turning_baseline_core")
    current, current_hash = load_core(current_path, "turning_current_core")
    before, after = run_matrix(baseline), run_matrix(current)
    result = dict(physics="deterministic lagged point-mass simulation; not AirSim or real flight",
                  limitations="No rotors, thrust/attitude dynamics, camera pitch, or hardware validation.",
                  baseline_source=str(args.baseline.resolve()), baseline_sha256=baseline_hash,
                  current_source=str(current_path), current_sha256=current_hash,
                  current_source_unchanged=hashlib.sha256(current_path.read_bytes()).hexdigest() == current_hash,
                  baseline_summary=summarize(before), current_summary=summarize(after),
                  baseline=before, current=after)
    result["passed"] = bool(result["current_source_unchanged"] and all(row["passed"] for row in after))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output),
                      "baseline_summary": result["baseline_summary"],
                      "current_summary": result["current_summary"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

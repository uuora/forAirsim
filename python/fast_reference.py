"""The ordinary forAirsim reference and TOPPQuad are checked by the same route proxy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from dual_target_control.route_curve import RouteCurve, decode_route, route_boundaries
from trajectory_research.curves import PolynomialSegment
from trajectory_research.route_evaluation import rollout_reference, sampled_route_metrics, route_derivative_bounds


GEOMETRIES = ("hermite", "bezier")
ARMS = ("ordinary", "toppquad", "toppquad-fixed")


def ordinary_plan(spec, geometry):
    if geometry not in GEOMETRIES:
        raise ValueError("未知轨迹构造")
    segments = []
    for bound in route_boundaries(spec):
        length = np.linalg.norm(np.asarray(bound["p1"]) - bound["p0"])
        duration = min(7., max(3., length/1.6 + 1.))
        segments.append(dict(parameters=[0.,0.,0.,float(duration)], boundary=bound))
    return dict(group_id=geometry+"_ordinary", representation=geometry, family="route_single_bump",
                seed=spec["seed"], route=spec, segments=segments, boundary=segments[0]["boundary"],
                parameters=segments[0]["parameters"])


def validate(curve, *, profile_model=None):
    limits = curve.spec["command_limits"]
    times = np.unique(np.r_[np.arange(0.,curve.duration_s,.02), curve.ends, curve.duration_s])
    q,v,a = [curve.sample(times, derivative=d) for d in range(3)]
    pos,vel,clip = rollout_reference(times,q,v,a,limits,curve.spec.get("tracking"))
    m = sampled_route_metrics(curve,times,pos,vel,clip,q,v)
    bounds = route_derivative_bounds(curve.segments,depth=2)
    feasible = m["control_error_gates_passed"] and all(x<=y for x,y in
        zip(bounds,(limits["max_speed_mps"],limits["max_slew_mps2"],30.)))
    result = dict(model_feasible=bool(feasible), derivative_bounds=list(bounds), metrics=m)
    if profile_model is not None:
        from segment_retiming import sampled_motor_forces, tracking_reserves
        margin = curve.spec["retiming"].get("tracking_error_margin_m", 0.)
        reserve = tracking_reserves(times, q, v, a, vel, curve.spec, m["max_position_error_m"], margin)
        motors = sampled_motor_forces(times, a, profile_model)
        result.update(tracking_reserves=reserve, sampled_fixed_yaw_motor_forces=motors)
        result["model_feasible"] &= reserve["peak_margin_passed"] and reserve["braking_passed"] and motors["passed"]
    return result


def decode(plan):
    if "timed_segments" not in plan:
        return decode_route(plan)
    pieces = [PolynomialSegment(x["coefficients"],x["duration_s"]) for x in plan["timed_segments"]]
    return RouteCurve(pieces,plan["waypoint_times"],plan["route"])


def solve_blocks(base, dynamics_model):
    from route_timing import solve_motion

    limits = base.spec["command_limits"]
    moving = list(base.segments[::2] if base.spec["policy"] == "stop" else base.segments)
    blocks = [[segment] for segment in moving] if base.spec["policy"] == "stop" else [moving]
    optimize_speed = (base.spec["policy"] == "pass"
                      and base.spec.get("retiming", {}).get("continuous_waypoint_speed") == "optimize")
    gains = (.9, .9, .9)
    if base.spec["policy"] == "pass" and "continuous_profile" in base.spec.get("retiming", {}):
        gains = base.spec.get("tracking", {}).get("acceleration_feedforward_s", gains)
    return [(block, solve_motion(block, limits["max_speed_mps"], limits["max_slew_mps2"], dynamics_model,
                                 fixed_waypoint_speed=not optimize_speed, feedforward_s=gains))
            for block in blocks]


def frozen_blocks(base_plan, source):
    raw = Path(source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("冻结计时源文件哈希不一致")
    previous = json.loads(raw)["plan"]
    previous_route = {key: value for key, value in previous["route"].items() if key != "retiming"}
    route = {key: value for key, value in base_plan["route"].items() if key != "retiming"}
    keys = ("representation", "segments", "dynamics_model")
    current = json.loads(json.dumps({key: base_plan[key] for key in keys}))
    if (route != previous_route or any(previous[key] != current[key] for key in keys)
            or route["policy"] != "pass" or len(previous["solver"]) != 1
            or previous["solver"][0].get("fixed_waypoint_speed") is not False):
        raise ValueError("冻结计时与当前连续任务或动力模型不一致")
    return [(list(decode_route(base_plan).segments), previous["solver"][0])]


def nominal_command_bounds(curve):
    tracking = curve.spec.get("tracking", {})
    feedforward = np.asarray(tracking.get("acceleration_feedforward_s", [.9] * 3))
    # Derivatives of p + K dp/dt are the nominal speed command and its slew.
    command_primitives = []
    for piece in curve.segments:
        coefficients = piece.coefficients.copy()
        if piece.degree:
            derivative = np.polynomial.polynomial.polyder(coefficients, axis=0) / piece.duration_s
            coefficients[:-1] += derivative * feedforward
        command_primitives.append(PolynomialSegment(coefficients, piece.duration_s))
    return route_derivative_bounds(command_primitives, depth=3)[:2]


def retimed(base_plan, blocks, fraction=None, *, profile=None):
    from route_timing import export_motion

    spec = base_plan["route"]
    stop = spec["policy"] == "stop"
    pieces, times, solutions = [], [], []
    for block, solution in blocks:
        # h=1 is baseline timing. At free via points the exported speed is
        # determined by the blended h rather than a waypoint speed equality.
        if profile is None:
            h = (1. + fraction * (np.asarray(solution["h"]) - 1.)).tolist()
            segments, end_indices = export_motion(block, dict(solution, h=h))
        else:
            segments, end_indices = export_motion(block, solution, time_map=profile)
        offset = len(pieces)
        pieces.extend(segments)
        # Use the same cumulative sum as RouteCurve so the last event is
        # exactly on the terminal sample, including after dwell segments.
        elapsed = np.cumsum([p.duration_s for p in pieces])
        for index in end_indices:
            i = len(times)
            arrival = float(elapsed[offset + index - 1])
            until = arrival
            if stop and i+1 < len(blocks):
                hold = PolynomialSegment([spec["waypoints"][i]],spec["dwell_s"])
                pieces.append(hold)
                until = float(np.cumsum([p.duration_s for p in pieces])[-1])
            speed = float(np.linalg.norm(segments[index - 1].sample(segments[index - 1].duration_s, derivative=1)))
            times.append(dict(index=i,arrival_time_s=arrival,evaluation_time_s=until,point=spec["waypoints"][i],
                              reference_speed_mps=speed))
        solutions.append(solution)
    plan = deepcopy(base_plan)
    plan.update(group_id=base_plan["representation"]+"_toppquad",solver=solutions,retiming_fraction=fraction,
                waypoint_times=times,timed_segments=[dict(coefficients=p.coefficients.tolist(),duration_s=p.duration_s) for p in pieces])
    if profile is not None:
        plan["retiming_profile"] = dict(mode="segmented", interpolation="quintic_smoothstep",
                                       phase_knots_s=profile.knots.tolist(), fractions=profile.fractions.tolist())
    return plan


def validate_timing(candidate, nominal_limits, *, enforce_reserve, check_profile):
    curve = decode(candidate)
    model = candidate["dynamics_model"] if check_profile else None
    validation = validate(curve, profile_model=model)
    bounds = nominal_command_bounds(curve)
    passed = all(bound <= limit for bound, limit in zip(bounds, nominal_limits))
    validation.update(nominal_command_bounds=bounds, nominal_command_limits=nominal_limits, nominal_reserve_passed=passed)
    if enforce_reserve:
        validation["model_feasible"] &= passed
    return curve.duration_s, validation


def select_segment_timing(base_plan, blocks, nominal_limits, seed_fractions):
    from segment_retiming import SegmentTimeMap, search_profiles

    (segments, solution), = blocks
    knots = np.r_[0., np.cumsum([piece.duration_s for piece in segments])]

    def make_candidate(fractions):
        return retimed(base_plan, blocks, profile=SegmentTimeMap(solution, knots, fractions))

    def evaluate(fractions):
        duration, validation = validate_timing(make_candidate(fractions), nominal_limits,
                                               enforce_reserve=True, check_profile=True)
        return dict(fractions=list(fractions), duration_s=duration, model_feasible=validation["model_feasible"],
                    validation=validation)

    best, trials = search_profiles(len(knots), seed_fractions, evaluate)
    # Keep the rejected baseline-shaped evaluation if no profile passes. The
    # execution owner persists the rejection and lands before reference start.
    selected = list(best) if best is not None else [0.] * len(knots)
    winner = next(row for row in trials if row["fractions"] == selected)
    return make_candidate(selected), winner["validation"], trials


def make_plan(output, spec, method, dynamics_model):
    geometry, arm = method.rsplit("_",1)
    if arm not in ARMS:
        raise ValueError("未知计时方法")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = perf_counter()
    plan = ordinary_plan(spec,geometry)
    plan["dynamics_model"] = dynamics_model
    candidates = []
    solves = 0
    timings = {}
    if arm != "ordinary":
        retiming = spec.get("retiming", {})
        source = retiming.get("frozen_source")
        raw_started = perf_counter()
        blocks = frozen_blocks(plan, source) if source else solve_blocks(decode_route(plan), dynamics_model)
        timings["frozen_source_read_wall_s" if source else "raw_toppquad_wall_s"] = perf_counter() - raw_started
        solves = 0 if source else len(blocks)
        selection_started = perf_counter()
        limits = spec["command_limits"]
        nominal_limits = [limits["max_speed_mps"] * retiming.get("nominal_speed_fraction", 1.),
                          limits["max_slew_mps2"] * retiming.get("nominal_slew_fraction", 1.)]
        fractions = retiming.get("fractions", [1.])
        segment_experiment = (spec["policy"] == "pass" and retiming.get("continuous_profile") == "segmented"
                              and source is None)
        if arm == "toppquad" and segment_experiment:
            plan, validation, candidates = select_segment_timing(plan, blocks, nominal_limits, fractions)
        else:
            for fraction in fractions:
                candidate = retimed(plan, blocks, fraction)
                duration, validation = validate_timing(candidate, nominal_limits,
                                                      enforce_reserve=bool(retiming and source is None),
                                                      check_profile=segment_experiment)
                candidates.append(dict(fraction=fraction, duration_s=duration, validation=validation))
                if validation["model_feasible"]:
                    break
            plan = candidate
        plan["group_id"] = method
        timings["retiming_selection_wall_s"] = perf_counter() - selection_started
    else:
        validation_started = perf_counter()
        validation = validate(decode(plan))
        timings["ordinary_validation_wall_s"] = perf_counter() - validation_started
    timings["search_plus_selected_validation_wall_s"] = perf_counter()-started
    plan["selection_provenance"] = dict(model_feasible=validation["model_feasible"],result_json="validation.json")
    evidence = dict(plan,counts=dict(selected_validation_calls=len(candidates) or 1,
                                   toppquad_solves=solves,
                                   airsim_search_calls=0),
                    timings=timings,validation=validation,retiming_candidates=candidates)
    (output/"validation.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    path = output/"airsim_replay_plan.json"
    path.write_text(json.dumps(dict(schema_version=1,plans=[plan]),ensure_ascii=False,allow_nan=False),encoding="utf8")
    return path

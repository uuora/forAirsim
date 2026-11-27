"""The ordinary forAirsim reference and TOPPQuad are checked by the same route proxy."""
from copy import deepcopy
import hashlib
import json
from math import factorial
from pathlib import Path
from time import perf_counter

import numpy as np

from dual_target_control.route_curve import RouteCurve, decode_route, distance_based_duration, route_boundaries
from dual_target_control.small_field import reference_checks
from trajectory_research.curves import PolynomialSegment
from trajectory_research.route_evaluation import rollout_reference, sampled_route_metrics, route_derivative_bounds


GEOMETRIES = ("hermite", "bezier", "bspline")
ARMS = ("ordinary", "toppquad", "toppquad-fixed", "toppquad-full")


def ordinary_plan(spec, geometry):
    if geometry not in GEOMETRIES:
        raise ValueError("未知轨迹构造")
    if geometry == "bspline":
        plan = bspline_plan(spec)
        return limit_reference_clock(plan) if "small_field" in spec else plan
    segments = []
    for bound in route_boundaries(spec):
        length = np.linalg.norm(np.asarray(bound["p1"]) - bound["p0"])
        duration = distance_based_duration(length)
        segments.append(dict(parameters=[0.,0.,0.,float(duration)], boundary=bound))
    plan = dict(group_id=geometry+"_ordinary", representation=geometry, family="route_single_bump",
                seed=spec["seed"], route=spec, segments=segments, boundary=segments[0]["boundary"],
                parameters=segments[0]["parameters"])
    if "small_field" in spec:
        return limit_reference_clock(plan)
    return plan


def limit_reference_clock(plan):
    curve = decode(plan)
    spec = curve.spec
    rules = spec["small_field"]
    extrema = reference_checks(curve)
    factor = max(1., extrema["peak_speed_mps"] / (0.99 * rules["reference_speed_mps"]),
                 np.sqrt(extrema["peak_acceleration_mps2"] / (0.99 * rules["reference_acceleration_mps2"])))
    stop = spec["policy"] == "stop"
    elapsed = 0.
    pieces, events = [], []
    for i, event in enumerate(curve.waypoint_times):
        segment = curve.segments[2*i if stop else i]
        duration = float(segment.duration_s * factor)
        pieces.append(dict(coefficients=segment.coefficients.tolist(), duration_s=duration))
        elapsed += duration
        arrival = elapsed
        if stop and i < len(curve.waypoint_times) - 1:
            dwell = spec["dwell_s"]
            pieces.append(dict(coefficients=[event["point"]], duration_s=dwell))
            elapsed += dwell
        speed = np.linalg.norm(segment.sample(segment.duration_s, derivative=1)) / factor
        events.append(dict(event, arrival_time_s=arrival, evaluation_time_s=elapsed, reference_speed_mps=float(speed)))
    return dict(plan, timed_segments=pieces, waypoint_times=events)


def bspline_plan(spec):
    from scipy.interpolate import make_interp_spline

    if spec["policy"] != "pass" or (spec.get("waypoint_acceptance") != "position_only" and "small_field" not in spec):
        raise ValueError("B 样条基线需要中间航点只验位置")
    points = np.asarray([spec["start"], *spec["waypoints"]])
    durations = [distance_based_duration(np.linalg.norm(d))/spec["time_scale"] for d in np.diff(points, axis=0)]
    times = np.r_[0., np.cumsum(durations)]
    rest = [(order, np.zeros(3)) for order in (1, 2, 3)]
    knots = np.r_[[times[0]]*8, times[1:-1], [times[-1]]*8]
    spline = make_interp_spline(times, points, k=7, t=knots, bc_type=(rest, rest))
    # Freeze normalized power polynomials, one per knot span, for the same runtime decoder.
    pieces = [dict(coefficients=[(spline(t, nu=d)*T**d/factorial(d)).tolist() for d in range(8)],
                   duration_s=T) for t,T in zip(times[:-1], durations)]
    events = [dict(index=i, point=point, arrival_time_s=t, evaluation_time_s=t,
                   reference_speed_mps=float(np.linalg.norm(spline(t, nu=1))))
              for i, (point,t) in enumerate(zip(spec["waypoints"], times[1:]))]
    return dict(group_id="bspline_ordinary", representation="bspline", family="clamped_septic_interpolation",
                seed=spec["seed"], route=spec, timed_segments=pieces, waypoint_times=events)


def validate(curve, *, profile_model=None):
    limits = curve.spec["command_limits"]
    times = np.unique(np.r_[np.arange(0.,curve.duration_s,.02), curve.ends, curve.duration_s])
    q,v,a = [curve.sample(times, derivative=d) for d in range(3)]
    pos,vel,clip = rollout_reference(times,q,v,a,limits,curve.spec.get("tracking"))
    m = sampled_route_metrics(curve,times,pos,vel,clip,q,v)
    if "small_field" in curve.spec:
        reference = reference_checks(curve)
        return dict(model_feasible=reference["passed"], reference_checks=reference, metrics=m)
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
    rules = base.spec.get("small_field", {})
    reference_limits = dict(reference_speed=rules["reference_speed_mps"],
                            reference_acceleration=rules["reference_acceleration_mps2"]) if rules else {}
    return [(block, solve_motion(block, limits["max_speed_mps"], limits["max_slew_mps2"], dynamics_model,
                                 fixed_waypoint_speed=not optimize_speed, feedforward_s=gains, **reference_limits))
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


def frozen_phase_timing(base_plan, blocks, profile):
    from scipy.interpolate import CubicSpline
    (segments, solution), = blocks
    knots = np.asarray(profile["knots"], dtype=float)
    rates_squared = np.asarray(profile["h"], dtype=float)
    if (knots.ndim != 1 or not 2 <= len(knots) <= 100 or rates_squared.shape != knots.shape
            or not np.isfinite(knots).all() or not np.isfinite(rates_squared).all()
            or knots[0] != solution["anchors"][0] or knots[-1] != solution["anchors"][-1]
            or np.any(np.diff(knots) <= 0)):
        raise ValueError("冻结相位计时节点无效")
    spline = CubicSpline(knots, rates_squared, bc_type="natural", extrapolate=False)
    stationary = spline.derivative().roots(extrapolate=False)
    if spline(np.r_[knots, stationary[np.isfinite(stationary)]]).min() < 1.e-6:
        raise ValueError("冻结相位计时必须保持正向推进")
    return retimed(base_plan, [(segments, solution)], phase_map=spline,
                   phase_profile=dict(mode="frozen_cubic", phase_knots_s=knots.tolist(),
                                      phase_rate_squared=rates_squared.tolist()))


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
    basis = curve.spec.get("retiming", {}).get("nominal_bound_basis", "bernstein")
    if basis == "minvo":
        from trajectory_research.minvo import control_norm_bound
        return route_derivative_bounds(command_primitives, depth=3, control_bound=control_norm_bound)[:2]
    if basis != "bernstein":
        raise ValueError("未知名义指令上界基函数")
    return route_derivative_bounds(command_primitives, depth=3)[:2]


def retimed(base_plan, blocks, fraction=None, *, profile=None, phase_map=None, phase_profile=None):
    from route_timing import export_motion

    spec = base_plan["route"]
    stop = spec["policy"] == "stop"
    pieces, times, solutions = [], [], []
    for block, solution in blocks:
        # h=1 is baseline timing. At free via points the exported speed is
        # determined by the blended h rather than a waypoint speed equality.
        if phase_map is not None:
            segments, end_indices = export_motion(block, solution, time_map=phase_map)
        elif profile is None:
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
    if "timed_segments" in base_plan:
        plan["base_timed_segments"] = plan["timed_segments"]
    plan.update(group_id=base_plan["representation"]+"_toppquad",solver=solutions,retiming_fraction=fraction,
                waypoint_times=times,timed_segments=[dict(coefficients=p.coefficients.tolist(),duration_s=p.duration_s) for p in pieces])
    if profile is not None:
        plan["retiming_profile"] = dict(mode="segmented", interpolation="quintic_smoothstep",
                                       phase_knots_s=profile.knots.tolist(), fractions=profile.fractions.tolist())
    if phase_profile is not None:
        plan["retiming_profile"] = phase_profile
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
    if geometry == "bspline" and spec.get("retiming", {}).get("frozen_source"):
        raise ValueError("B 样条暂不支持旧版冻结比例扫描入口")
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
        blocks = frozen_blocks(plan, source) if source else solve_blocks(decode(plan), dynamics_model)
        timings["frozen_source_read_wall_s" if source else "raw_toppquad_wall_s"] = perf_counter() - raw_started
        solves = 0 if source else len(blocks)
        selection_started = perf_counter()
        limits = spec["command_limits"]
        nominal_limits = [limits["max_speed_mps"] * retiming.get("nominal_speed_fraction", 1.),
                          limits["max_slew_mps2"] * retiming.get("nominal_slew_fraction", 1.)]
        fractions = [1.] if arm == "toppquad-full" else retiming.get("fractions", [1.])
        segment_experiment = (spec["policy"] == "pass" and retiming.get("continuous_profile") == "segmented"
                              and source is None)
        phase_profile = retiming.get("frozen_phase_profile")
        if phase_profile is not None:
            if not source or arm != "toppquad" or spec.get("waypoint_acceptance") != "position_only":
                raise ValueError("冻结相位计时需要冻结 TOPPQuad 连续过点方案")
            plan = frozen_phase_timing(plan, blocks, phase_profile)
            _, validation = validate_timing(plan, nominal_limits, enforce_reserve=True, check_profile=True)
        elif arm == "toppquad" and segment_experiment:
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

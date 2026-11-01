"""Session-held-out identification of a historical AirSim velocity-API proxy.

The input logs contain host tick times, a state read followed by a command,
and no simulator/send/ack timestamps. Consequently this script estimates only
an EFFECTIVE low-order velocity-API closed-loop predictor under that sampling.
Its fitted lag is not a physical aircraft/actuator time constant. It does not
connect to AirSim, change the production controller, or conduct new flights.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "logs/dual_target_airsim_20261015_195940_290437"
VALIDATION = ROOT / "logs/dual_target_airsim_20261015_200705_681433"
DEFAULT_RECEIPTS = (
    ROOT / "reports/dual_target_control/short_waypoints_twr_20261015/validation_analysis_inbound90.json",
    ROOT / "reports/dual_target_control/short_waypoints_twr_20261015/validation_analysis.json",
)
DELAYS_S = (0., .01, .02, .04, .06)
MAX_DT_S = .15
DEFAULT_TAU_S = .30
TAU_GRID = np.geomspace(.02, 2., 301)
SCOPE = (
    "Historical AirSim SimpleFlight velocity-API closed-loop effective proxy; "
    "host-time sampling, not physical aircraft/actuator identification. "
    "No simulator, send, effective-input or acknowledgement timestamps; "
    "no attitude, thrust, motor or wind identification. No new flights."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_name(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def finite_scalar(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)) \
        and np.isrealobj(value) and math.isfinite(float(value))


def finite_vector(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.shape == (3,) and array.dtype.kind in "iuf" and bool(np.isfinite(array).all())


def statistics(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return dict(n=0, min=None, median=None, p95=None, max=None, mean=None)
    return dict(n=len(array), min=float(np.min(array)), median=float(np.median(array)),
                p95=float(np.quantile(array, .95)), max=float(np.max(array)), mean=float(np.mean(array)))


@dataclass
class Session:
    name: str
    directory: Path
    rows: list
    segments: list
    row_rejections: list
    transition_rejections: list
    source_files: dict
    summary_identity: dict
    nominal_speed_cap_mps: float | None


def load_session(directory, max_dt=MAX_DT_S):
    """Preserve line identity and never form transitions across invalid gaps."""
    directory = Path(directory).resolve()
    telemetry = directory / "telemetry.jsonl"
    summary = read_json(directory / "summary.json")
    if summary.get("mode") != "airsim" or "AirSim" not in str(summary.get("physics", "")):
        raise ValueError("input session must explicitly identify actual AirSim mode/physics")
    config = read_json(directory / "config.json") if (directory / "config.json").exists() else {}
    speed = config.get("control", {}).get("max_speed")
    cap = float(speed) if finite_scalar(speed) and speed > 0 else None
    rows, row_rejections = [], []
    for index, line in enumerate(telemetry.read_text(encoding="utf-8-sig").splitlines()):
        reasons = []
        try:
            raw = json.loads(line)
        except (ValueError, TypeError):
            raw = None
        if not isinstance(raw, dict):
            reasons.append("invalid_json_object")
        else:
            if not finite_scalar(raw.get("t")):
                reasons.append("invalid_host_time")
            for key in ("position", "velocity", "velocity_command"):
                if not finite_vector(raw.get(key)):
                    reasons.append("invalid_" + key)
        if reasons:
            rows.append(None)
            row_rejections.append(dict(row_index=index, source_line=index + 1, reasons=reasons))
            continue
        rows.append(dict(row_index=index, source_line=index + 1, t=float(raw["t"]),
            position=np.asarray(raw["position"], dtype=float),
            velocity=np.asarray(raw["velocity"], dtype=float),
            command=np.asarray(raw["velocity_command"], dtype=float),
            rpc_ms=float(raw["rpc_ms"]) if finite_scalar(raw.get("rpc_ms")) else None,
            loop_ms=float(raw["loop_ms"]) if finite_scalar(raw.get("loop_ms")) else None,
            interval_ms=float(raw["interval_ms"]) if finite_scalar(raw.get("interval_ms")) else None))
    segments, current, rejected = [], [], []
    for index, row in enumerate(rows):
        if row is None:
            if current:
                segments.append(current)
                current = []
        else:
            if index and rows[index - 1] is not None:
                dt = row["t"] - rows[index - 1]["t"]
                if not 0 < dt <= max_dt:
                    if current:
                        segments.append(current)
                    current = []
            current.append(index)
    if current:
        segments.append(current)
    for index in range(max(0, len(rows) - 1)):
        first, second = rows[index:index + 2]
        if first is None or second is None:
            reasons, dt = ["transition_touches_invalid_row"], None
        else:
            dt = second["t"] - first["t"]
            reasons = (["nonpositive_host_dt"] if dt <= 0 else
                       ["host_dt_exceeds_maximum"] if dt > max_dt else [])
        if reasons:
            rejected.append(dict(from_row_index=index, to_row_index=index + 1,
                from_source_line=index + 1, to_source_line=index + 2, dt_s=dt, reasons=reasons))
    files = {name: dict(path=source_name(directory / name), sha256=digest(directory / name),
                       bytes=(directory / name).stat().st_size)
             for name in ("telemetry.jsonl", "summary.json", "config.json", "dynamics.json", "validation.json")
             if (directory / name).is_file()}
    return Session(directory.name, directory, rows, segments, row_rejections, rejected, files,
        dict(mode=summary["mode"], physics=summary["physics"], outcome=summary.get("outcome"),
             declared_samples=summary.get("samples"), hits=summary.get("hits")), cap)


def session_statistics(session):
    rows = [row for row in session.rows if row is not None]
    transition_rows = [(session.rows[a], session.rows[b]) for segment in session.segments
                       for a, b in zip(segment[:-1], segment[1:])]
    velocity = np.array([row["velocity"] for row in rows])
    commands = np.array([row["command"] for row in rows])
    speeds = np.linalg.norm(velocity, axis=1)
    command_speeds = np.linalg.norm(commands, axis=1)
    dt = [b["t"] - a["t"] for a, b in transition_rows]
    discrepancy = [abs((b["t"] - a["t"]) - b["interval_ms"] / 1000.)
                   for a, b in transition_rows if b["interval_ms"] is not None]
    command_delta = np.array([b["command"] - a["command"] for a, b in transition_rows])
    cap = session.nominal_speed_cap_mps
    return dict(raw_rows=len(session.rows), finite_rows=len(rows), monotone_segments=len(session.segments),
        retained_base_transitions=len(dt), rejected_rows=len(session.row_rejections),
        rejected_transitions=len(session.transition_rejections),
        time_range_s=[rows[0]["t"], rows[-1]["t"]], actual_host_dt_s=statistics(dt),
        declared_interval_vs_host_dt_absolute_difference_s=statistics(discrepancy),
        rpc_ms=statistics([r["rpc_ms"] for r in rows if r["rpc_ms"] is not None]),
        loop_ms=statistics([r["loop_ms"] for r in rows if r["loop_ms"] is not None]),
        command_per_axis_min_mps=np.min(commands, axis=0).tolist(),
        command_per_axis_max_mps=np.max(commands, axis=0).tolist(),
        command_per_axis_std_mps=np.std(commands, axis=0).tolist(),
        command_transition_delta_rms_mps=np.sqrt(np.mean(command_delta ** 2, axis=0)).tolist(),
        observed_speed_mps=statistics(speeds), command_speed_mps=statistics(command_speeds),
        nominal_command_speed_cap_mps=cap,
        observed_speed_above_nominal_cap_fraction=float(np.mean(speeds > cap + 1e-9)) if cap else None,
        command_speed_above_nominal_cap_fraction=float(np.mean(command_speeds > cap + 1e-9)) if cap else None,
        overshoot_interpretation="Measured velocity relative to nominal command cap; not an identified actuator/flight-safety limit")


@dataclass
class Prepared:
    delay_s: float
    indices: np.ndarray
    dt: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    initial_command: np.ndarray
    remaining_start: np.ndarray
    remaining_end: np.ndarray
    piece_commands: np.ndarray
    pieces: list
    rejected: list


def prepare_transitions(session, delay):
    """Build exact ZOH pieces for the ASSUMED host-timed delayed command.

    Delayed command changes at historical row_time+delay may lie inside a
    transition. They are integrated piecewise, rather than silently freezing
    c(t_k-delay) over an interval containing a known input change.
    """
    if not finite_scalar(delay) or delay < 0:
        raise ValueError("delay must be finite and nonnegative")
    items, rejected = [], []
    for segment in session.segments:
        times = np.array([session.rows[i]["t"] for i in segment])
        for k in range(len(segment) - 1):
            first, second = session.rows[segment[k]], session.rows[segment[k + 1]]
            begin, end = first["t"], second["t"]
            j = int(np.searchsorted(times, begin - delay, side="right") - 1)
            if j < 0:
                rejected.append(dict(from_row_index=first["row_index"], to_row_index=second["row_index"],
                                     reason="delay_history_unavailable", delay_s=delay))
                continue
            pieces, current = [], begin
            initial_command = session.rows[segment[j]]["command"]
            while current < end:
                next_change = times[j + 1] + delay if j + 1 < len(times) else math.inf
                piece_end = min(end, float(next_change))
                if piece_end <= current:
                    j += 1
                    continue
                pieces.append(dict(start_s=current, end_s=piece_end,
                    command_row_index=segment[j], command=session.rows[segment[j]]["command"]))
                current = piece_end
                if current < end:
                    j += 1
            items.append((first, second, initial_command, pieces))
    if not items:
        raise ValueError("no retained transitions at the declared delay")
    width = max(len(item[3]) for item in items)
    remaining_start = np.zeros((len(items), width))
    remaining_end = np.zeros_like(remaining_start)
    commands = np.zeros((len(items), width, 3))
    for i, (_, second, _, pieces) in enumerate(items):
        for j, piece in enumerate(pieces):
            remaining_start[i, j] = second["t"] - piece["start_s"]
            remaining_end[i, j] = second["t"] - piece["end_s"]
            commands[i, j] = piece["command"]
    return Prepared(float(delay), np.array([x[0]["row_index"] for x in items], dtype=int),
        np.array([x[1]["t"] - x[0]["t"] for x in items]),
        np.array([x[0]["velocity"] for x in items]), np.array([x[1]["velocity"] for x in items]),
        np.array([x[2] for x in items]), remaining_start, remaining_end, commands,
        [x[3] for x in items], rejected)


def predict_lag(prepared, tau, velocity=None):
    if not finite_scalar(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    initial = prepared.velocity if velocity is None else np.asarray(velocity, dtype=float)
    if initial.shape != prepared.velocity.shape or not np.isfinite(initial).all():
        raise ValueError("initial velocity matrix shape/finiteness mismatch")
    weights = np.exp(-prepared.remaining_end / tau) - np.exp(-prepared.remaining_start / tau)
    return np.exp(-prepared.dt / tau)[:, None] * initial \
        + np.sum(weights[:, :, None] * prepared.piece_commands, axis=1)


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    if error.ndim != 2 or error.shape[1] != 3 or not len(error) or not np.isfinite(error).all():
        raise ValueError("three-dimensional finite prediction residuals required")
    norms = np.linalg.norm(error, axis=1)
    return dict(samples=len(error), vector_rmse_mps=float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        per_axis_rmse_mps=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        vector_mean_error_mps=np.mean(error, axis=0).tolist(),
        vector_absolute_error_p95_mps=float(np.quantile(norms, .95)))


def fit_tau(prepared, grid=TAU_GRID):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(grid <= 0):
        raise ValueError("a finite positive tau grid is required")
    mse = np.array([np.mean(np.sum((predict_lag(prepared, tau) - prepared.target) ** 2, axis=1))
                    for tau in grid])
    selected = int(np.argmin(mse))
    return dict(tau_s=float(grid[selected]), grid_index=selected,
                selected_on_grid_boundary=selected in (0, len(grid) - 1),
                train_vector_mse_m2_s2=float(mse[selected]),
                grid_tau_s=grid.tolist(), grid_train_vector_mse_m2_s2=mse.tolist())


def model_predictions(prepared, fitted_tau):
    return dict(persistence=prepared.velocity,
        instant_command=prepared.initial_command,
        default_lag_0p30=predict_lag(prepared, DEFAULT_TAU_S),
        train_fitted_lag=predict_lag(prepared, fitted_tau))


def free_run_metrics(session, prepared, tau, requested_horizon):
    """Nonoverlapping measured-start forecasts, with no intermediate state reset.

    Endpoints use the first measured row at/after the requested horizon;
    actual durations are reported. Logged commands are used, not a rerun of
    the feedback controller; these are held-input velocity forecasts only.
    """
    by_index = {int(index): j for j, index in enumerate(prepared.indices)}
    predicted = {name: [] for name in ("persistence", "instant_command", "default_lag_0p30", "train_fitted_lag")}
    actual, horizons, windows = [], [], []
    for segment in session.segments:
        times = np.array([session.rows[i]["t"] for i in segment])
        start = 0
        while start < len(segment) - 1:
            end = int(np.searchsorted(times, times[start] + requested_horizon, side="left"))
            if end >= len(segment):
                break
            indices = segment[start:end]
            if not all(index in by_index for index in indices):
                start += 1
                continue
            first, last = session.rows[segment[start]], session.rows[segment[end]]
            states = {"default_lag_0p30": first["velocity"].copy(), "train_fitted_lag": first["velocity"].copy()}
            for index in indices:
                k = by_index[index]
                for name, parameter in (("default_lag_0p30", DEFAULT_TAU_S), ("train_fitted_lag", tau)):
                    v = states[name]
                    for piece in prepared.pieces[k]:
                        h = piece["end_s"] - piece["start_s"]
                        alpha = -math.expm1(-h / parameter)
                        v = v + (piece["command"] - v) * alpha
                    states[name] = v
            predicted["persistence"].append(first["velocity"])
            predicted["instant_command"].append(prepared.pieces[by_index[indices[-1]]][-1]["command"])
            for name, state in states.items():
                predicted[name].append(state)
            actual.append(last["velocity"])
            horizons.append(last["t"] - first["t"])
            windows.append(dict(start_row_index=first["row_index"], end_row_index=last["row_index"],
                                actual_horizon_s=last["t"] - first["t"]))
            start = end
    return dict(requested_horizon_s=requested_horizon, actual_horizon_s=statistics(horizons),
                windows=windows, metrics={name: metrics(actual, value) for name, value in predicted.items()} if actual else {},
                scope="Measured state only at window start; logged inputs, no controller resimulation or position-tracking validation")


def receipt_observations(sessions, paths):
    observations = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            observations.append(dict(path=source_name(path), status="UNAVAILABLE"))
            continue
        receipt = read_json(path)
        matches = []
        for session in sessions:
            for run in receipt.get("runs", {}).values():
                for name, expected in run.get("file_sha256", {}).items():
                    if str(name).replace("\\", "/").endswith(session.name + "/telemetry.jsonl"):
                        actual = session.source_files["telemetry.jsonl"]["sha256"]
                        if actual != expected:
                            raise ValueError("historical receipt telemetry hash mismatch")
                        matches.append(dict(session=session.name, expected_sha256=expected, status="MATCHED"))
        observations.append(dict(path=source_name(path), sha256=digest(path), status="READ", telemetry_matches=matches))
    return observations


def export_samples(session, path):
    segment_ids = {index: segment_id for segment_id, segment in enumerate(session.segments) for index in segment}
    fields = ["row_index", "source_line", "segment_id", "t_s", "world_x_m", "world_y_m", "height_m",
              "vx_mps", "vy_mps", "height_rate_mps", "command_x_mps", "command_y_mps", "command_height_mps",
              "rpc_ms", "loop_ms", "declared_interval_ms"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in session.rows:
            if row is not None:
                writer.writerow([row["row_index"], row["source_line"], segment_ids[row["row_index"]], row["t"],
                    *row["position"], *row["velocity"], *row["command"], row["rpc_ms"], row["loop_ms"], row["interval_ms"]])


def export_predictions(session, prepared, predictions, path):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        names = list(predictions)
        writer.writerow(["from_row_index", "to_row_index", "dt_s", "delay_s", "target_vx_mps", "target_vy_mps", "target_vheight_mps"]
                        + [f"{name}_{axis}_mps" for name in names for axis in ("vx", "vy", "vheight")])
        for i, index in enumerate(prepared.indices):
            writer.writerow([int(index), int(index) + 1, prepared.dt[i], prepared.delay_s, *prepared.target[i],
                             *(component for name in names for component in predictions[name][i])])


def interpret(summary):
    primary = summary["delay_sensitivity"][0]
    tau = primary["fit"]["tau_s"]
    test = primary["validation_metrics"]
    fitted, persistent = test["train_fitted_lag"]["vector_rmse_mps"], test["persistence"]["vector_rmse_mps"]
    default = test["default_lag_0p30"]["vector_rmse_mps"]
    relation = "lower" if fitted < persistent else "higher or equal"
    return (
        "# Historical AirSim velocity-API proxy identification\n\n"
        "AI assisted this script and interpretation. No new simulator session or flight was performed.\n\n"
        f"The zero-delay effective lag selected on the training session is {tau:.6g} s. "
        f"On the separate held-out session its one-step velocity-vector RMSE is {fitted:.6g} m/s, "
        f"versus persistence {persistent:.6g} m/s and the declared 0.30 s proxy {default:.6g} m/s. "
        f"The fitted model's error is {relation} than persistence.\n\n"
        "The parameter describes a sampled velocity API plus its internal controller and the observed operating conditions. "
        "It is not an identified aircraft or motor time constant. State is read before the same-row command is sent: "
        "the command is used to predict the next state. Host tick time is used without inventing simulator or dispatch timestamps. "
        "The declared delays are sensitivity cases, each fitted only on training data; validation is never used to select a delay.\n\n"
        "One-step errors can favor persistence at high sampling rates. The separately reported 0.5/1 s open-loop-input "
        "forecasts start from measured velocity once per window and do not reset to measured intermediate states. "
        "They do not rerun the feedback controller or validate position tracking.\n\n"
        "Commands are endogenous feedback inputs, two sessions do not establish broad generalization, and omitted attitude, "
        "input timing, overshoot, nonlinear coupling and delays can invalidate a one-pole proxy. Grid-boundary solutions, "
        "weak excitation, residuals and model-baseline comparisons must be inspected before reusing the fitted value. "
        "The existing trajectory study is not automatically retuned from these diagnostics.\n"
    )


def run(args):
    if Path(args.train).resolve() == Path(args.validation).resolve():
        raise ValueError("training and validation must be different whole sessions")
    started = time.perf_counter()
    train, validation = load_session(args.train), load_session(args.validation)
    if train.source_files["telemetry.jsonl"]["sha256"] == validation.source_files["telemetry.jsonl"]["sha256"]:
        raise ValueError("training and validation telemetry bytes must differ")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    shutil.copy2(__file__, snapshots / Path(__file__).name)
    manifest = dict(protocol="historical-airsim-effective-velocity-proxy-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope=SCOPE, python=platform.python_version(), numpy=np.__version__, source_sha256=digest(__file__),
        train=dict(session=train.name, source_files=train.source_files, summary_identity=train.summary_identity),
        validation=dict(session=validation.name, source_files=validation.source_files, summary_identity=validation.summary_identity),
        provenance_receipts=receipt_observations((train, validation), args.receipts),
        coordinate_system="world x/y and positive height z; command and measured velocity in the same frame, not raw NED",
        primary=dict(delay_s=0., tau_grid_s=TAU_GRID.tolist(), objective="training mean squared velocity-vector residual", axis_weights=[1., 1., 1.]),
        delay_sensitivity_s=list(DELAYS_S), default_lag_s=DEFAULT_TAU_S,
        sample_policy=dict(max_host_dt_s=MAX_DT_S, transitions="adjacent source rows only; finite and strictly increasing host times; no crossing rejected rows/gaps"),
        input_alignment="state v_k read before command c_k is generated/sent; predict v_(k+1) from v_k and assumed held command history",
        delayed_input="historical command ZOH at t-delay, with exact piecewise integration of known command changes inside each transition",
        selection="tau selected only on training session; zero delay primary; no held-out delay/model selection",
        free_run_horizons_s=[] if args.skip_free_run else [.5, 1.],
        limitations=["No per-row simulator/send/ack/input-effective time or pause flag", "Host time is not proved equal to simulator time", "Closed-loop endogenous commands, not designed system-identification excitation", "No physical tau, actuator saturation, thrust, wind or robust-flight conclusion"])
    # This protocol/data/source record is written BEFORE preparing or fitting models.
    write_json(output / "manifest.json", manifest)
    summaries = dict(scope=SCOPE, train_statistics=session_statistics(train), validation_statistics=session_statistics(validation), delay_sensitivity=[])
    prepared = {role: {delay: prepare_transitions(session, delay) for delay in DELAYS_S}
                for role, session in (("train", train), ("validation", validation))}
    common = {role: sorted(set.intersection(*(set(p.indices.tolist()) for p in by_delay.values())))
              for role, by_delay in prepared.items()}
    for role, session in (("train", train), ("validation", validation)):
        export_samples(session, output / f"{role}_numeric_samples.csv")
        write_json(output / f"{role}_rejections.json", dict(row_rejections=session.row_rejections,
            transition_rejections=session.transition_rejections,
            delay_rejections={str(delay): prepared[role][delay].rejected for delay in DELAYS_S}))
    for delay in DELAYS_S:
        fit = fit_tau(prepared["train"][delay])
        parameter = fit["tau_s"]
        group = dict(delay_s=delay, fit=fit, train_metrics={}, validation_metrics={}, matched_common_support_metrics={})
        for role, session in (("train", train), ("validation", validation)):
            item = prepared[role][delay]
            predictions = model_predictions(item, parameter)
            group[role + "_metrics"] = {name: metrics(item.target, pred) for name, pred in predictions.items()}
            keep = np.isin(item.indices, common[role])
            group["matched_common_support_metrics"][role] = {name: metrics(item.target[keep], pred[keep]) for name, pred in predictions.items()}
            export_predictions(session, item, predictions, output / f"{role}_delay_{delay:.2f}_predictions.csv")
        summaries["delay_sensitivity"].append(group)
    if not args.skip_free_run:
        tau = summaries["delay_sensitivity"][0]["fit"]["tau_s"]
        summaries["primary_zero_delay_free_run_forecasts"] = {
            role: [free_run_metrics(session, prepared[role][0.], tau, horizon) for horizon in (.5, 1.)]
            for role, session in (("train", train), ("validation", validation))}
    summaries["processing_wall_s"] = time.perf_counter() - started
    summaries["validation_used_for_selection"] = False
    write_json(output / "summary.json", summaries)
    (output / "interpretation.md").write_text(interpret(summaries), encoding="utf-8")
    for session in (train, validation):
        assert digest(session.directory / "telemetry.jsonl") == session.source_files["telemetry.jsonl"]["sha256"], "source telemetry changed during identification"
    assert digest(__file__) == manifest["source_sha256"], "script source changed during identification"
    output_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "completion.json"}
    write_json(output / "completion.json", dict(status="COMPLETE", validation_used_for_selection=False,
        source_bytes_unchanged=True, files_sha256=output_hashes, processing_wall_s=summaries["processing_wall_s"]))
    primary = summaries["delay_sensitivity"][0]
    print(json.dumps(dict(output=str(output), status="COMPLETE", fitted_effective_tau_s=primary["fit"]["tau_s"],
        primary_validation_metrics=primary["validation_metrics"], scope=SCOPE), ensure_ascii=False, allow_nan=False), flush=True)


def self_test():
    tau = float(TAU_GRID[123])
    dt = np.array([.011, .025, .018, .031, .014, .029])
    velocity = np.array([[.2, -.3, .5], [.6, -.2, .1], [-.2, .1, .3], [.4, .2, -.1], [.1, -.5, .2], [.7, .1, -.2]])
    commands = np.array([[1., -.5, .1], [.5, .2, -.1], [-.1, .8, .2], [.9, -.2, .4], [.5, .5, .7], [.1, -.1, .3]])
    target = np.exp(-dt / tau)[:, None] * velocity + (-np.expm1(-dt / tau))[:, None] * commands
    item = Prepared(0., np.arange(len(dt)), dt, velocity, target, commands, dt[:, None], np.zeros((len(dt), 1)), commands[:, None, :], [], [])
    np.testing.assert_allclose(predict_lag(item, tau), target, rtol=1e-14, atol=1e-14)
    assert fit_tau(item)["grid_index"] == 123
    # Two held-input pieces must match a separate sequential lag recurrence.
    v0, c0, c1 = np.array([[.4, -.2, .1]]), np.array([1., .2, -.3]), np.array([-.1, .8, .5])
    h0, h1 = .012, .021
    exact = (v0[0] + (c0 - v0[0]) * (-math.expm1(-h0 / tau)))
    exact = exact + (c1 - exact) * (-math.expm1(-h1 / tau))
    pieces = Prepared(.01, np.array([0]), np.array([h0 + h1]), v0, exact[None, :], c0[None, :],
        np.array([[h0 + h1, h1]]), np.array([[h1, 0.]]), np.array([[c0, c1]]), [], [])
    np.testing.assert_allclose(predict_lag(pieces, tau)[0], exact, rtol=1e-14, atol=1e-14)
    assert metrics(target, target)["vector_rmse_mps"] == 0.
    print("PASS: irregular-step exact lag, synthetic grid recovery, piecewise delayed-input integration, residual metric")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(TRAIN))
    parser.add_argument("--validation", default=str(VALIDATION))
    parser.add_argument("--receipts", nargs="*", default=[str(p) for p in DEFAULT_RECEIPTS])
    parser.add_argument("--output")
    parser.add_argument("--skip-free-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.output is None:
        parser.error("--output is required; use a new directory")
    else:
        run(arguments)

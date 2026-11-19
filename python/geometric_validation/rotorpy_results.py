"""Recalculate RotorPy records and build an offline trajectory browser."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np

from .reference import FrozenReference
from .runtime import ROOT, digest, write_json


def load(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def audit_samples(folder, row, design, reference):
    data = np.genfromtxt(folder / "simulation.csv.gz", delimiter=",", names=True, encoding="utf-8")
    data = np.atleast_1d(data)
    assert len(data) == row["samples"]
    t = data["time_s"]
    np.testing.assert_allclose(t, np.arange(len(data)) / design["hz"], atol=1e-8, rtol=0)
    actual = np.column_stack([data[f"position_{axis}_m"] for axis in "enu"])
    target = np.column_stack([data[f"reference_{axis}_m"] for axis in "enu"])
    finite = np.logical_and.reduce([np.isfinite(data[name]) for name in data.dtype.names])
    if not finite.any():
        assert not row["accepted"]
        return None
    position_error = np.linalg.norm(actual[finite] - target[finite], axis=1)
    actual_q = np.column_stack([data["q" + axis] for axis in "xyzw"])[finite]
    reference_q = np.column_stack([data["reference_q" + axis] for axis in "xyzw"])[finite]
    cosine = np.abs(np.sum(actual_q * reference_q, axis=1))
    cosine /= np.linalg.norm(actual_q, axis=1) * np.linalg.norm(reference_q, axis=1)
    attitude = np.degrees(2 * np.arccos(np.clip(cosine, 0, 1)))
    np.testing.assert_allclose(position_error, data["position_error_m"][finite], atol=1e-7, rtol=0)
    np.testing.assert_allclose(attitude, data["attitude_error_deg"][finite], atol=2e-5, rtol=0)
    assert abs(float(position_error.max()) - row["position_peak_m"]) < 1e-7
    assert abs(float(np.sqrt(np.mean(position_error**2))) - row["position_rmse_m"]) < 1e-7
    assert abs(float(attitude.max()) - row["attitude_peak_deg"]) < 2e-5

    rule = design["passage"]
    points = np.asarray(reference["points"])[:, [1, 0, 2]] * [1, 1, -1]
    velocity = np.column_stack([data[f"velocity_{axis}_mps"] for axis in "enu"])
    speed = np.linalg.norm(velocity, axis=1)
    previous = -np.inf
    passed = True
    for point, knot, evidence in zip(points[1:-1], np.cumsum(reference["segment_durations"])[:-1], row["passages"], strict=True):
        distance = np.linalg.norm(actual - point, axis=1)
        window = finite & (t > previous) & (np.abs(t - knot) <= rule["reference_time_window_s"])
        match = np.flatnonzero(window & (distance <= rule["position_tolerance_m"])
                              & (speed >= rule["min_actual_speed_mps"]))
        if window.any():
            nearest = np.flatnonzero(window)[np.argmin(distance[window])]
            assert abs(distance[nearest] - evidence["nearest_distance_m"]) < 1e-7
            assert abs(speed[nearest] - evidence["nearest_speed_mps"]) < 1e-7
        else:
            assert evidence["nearest_distance_m"] is None
        assert evidence["passed"] == bool(len(match))
        if len(match):
            first = match[0]
            assert abs(t[first] - evidence["actual_time_s"]) < 1e-8
            assert abs(distance[first] - evidence["distance_m"]) < 1e-7
            previous = t[first]
        else:
            previous = np.inf
            passed = False
    assert passed == row["waypoint_passed"]
    limits = design["limits"]
    if row["sim_status"] == "completed":
        assert row["complete_s"] == row["last_sample_s"]
        held = (t + 1e-8 >= t[-1] - design["terminal_hold_s"])
        assert np.all(t[held] + 1e-8 >= row["reference_s"])
        assert np.all(data["position_error_m"][held] <= limits["terminal_position_m"] + 1e-7)
        assert np.all(data["attitude_error_deg"][held] <= np.degrees(limits["terminal_attitude_rad"]) + 2e-5)
        assert np.all(speed[held] <= limits["terminal_speed_mps"] + 1e-7)
    expected_gate = (row["sim_status"] == "completed" and passed
                     and position_error.max() <= limits["position_error_m"]
                     and attitude.max() <= np.degrees(limits["attitude_error_rad"])
                     and data["actual_tilt_deg"][finite].max() <= np.degrees(limits["safety_tilt_rad"]))
    assert row["gate_passed"] == expected_gate
    assert row["accepted"] == (row["gate_passed"] and not row["plan_issues"])
    keep = np.unique(np.r_[np.arange(0, len(t), max(1, len(t) // 180)), len(t) - 1])
    keep = keep[finite[keep]]
    return dict(actual=np.round(actual[keep], 5).tolist(), ref=np.round(target[keep], 5).tolist())


def audit_reference_matches(baseline, retimed):
    before, after = FrozenReference(baseline), FrozenReference(retimed)
    warp = np.polynomial.Polynomial(retimed["planning"]["progress_polynomial"])
    for fraction in np.linspace(0, 1, 100):
        at = after.sample(fraction * after.duration)["p"]
        on_path = before.sample(float(np.clip(warp(fraction), 0, 1)) * before.duration)["p"]
        np.testing.assert_allclose(at, on_path, atol=1e-7, rtol=0)


def export_browser(root, pairs):
    choices = "".join(f'<option value="{key}">{key.replace("/", " · ")}</option>' for key in pairs)
    assets = ROOT / "python/geometric_validation/assets/plotly-gl3d-3.1.0.min.js"
    shutil.copyfile(assets, root / assets.name)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>RotorPy 逐轨迹查看</title>
<style>body{font:16px system-ui;margin:24px;color:#19324a}#plot{height:70vh;min-height:400px}
select{padding:6px}a{color:#006980}</style>
<h1>RotorPy 逐轨迹查看</h1>
<p><a href="index.html">返回数据索引</a>　航点与轨迹使用 ENU，单位为米。原始 50 Hz 样本见逐次数据。</p>
<label>路线与基线 <select id="pair">OPTIONS</select></label>
<p id="outcome" aria-live="polite"></p><div id="plot"></div>
<script src="plotly-gl3d-3.1.0.min.js"></script>
<script>
const picker = document.getElementById('pair');
const outcome = document.getElementById('outcome');
let loaded;
let generation = 0;
const trace = (points, name, color, mode, dash) => ({
  type:'scatter3d', mode, name,
  x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2]),
  line:{color,dash,width:5}, marker:{color,size:4}
});
function update() {
  const request = ++generation;
  if (loaded) loaded.remove();
  outcome.textContent = '读取轨迹…';
  const key = picker.value;
  loaded = document.createElement('script');
  loaded.src = key + '/preview.js';
  loaded.onload = () => {
    if (request !== generation) return;
    const data = window.rotorpyPair;
    const traces = [trace(data.points, '航点', '#222d3c', 'markers')];
    const labels = {baseline:'原始基线', toppquad:'TOPPQuad'};
    const colors = {baseline:'#007789', toppquad:'#a33e84'};
    outcome.textContent = Object.entries(data.arms).map(([k,v]) =>
      labels[k] + ' ' + (v.accepted ? '通过' : '未通过') +
      (v.complete_s == null ? '' : ' · ' + v.complete_s.toFixed(2) + ' s')).join('　');
    for (const [arm, item] of Object.entries(data.arms)) {
      if (!item.curves) continue;
      traces.push(trace(item.curves.ref,labels[arm]+'参考',colors[arm],'lines','dot'));
      traces.push(trace(item.curves.actual,labels[arm]+'实测',colors[arm],'lines','solid'));
    }
    Plotly.react('plot',traces,{margin:{l:0,r:0,t:0,b:0},
      scene:{aspectmode:'data',xaxis:{title:'东 E (m)'},yaxis:{title:'北 N (m)'},zaxis:{title:'上 U (m)'}}},
      {responsive:true,displaylogo:false});
  };
  loaded.onerror = () => { if (request === generation) outcome.textContent = '轨迹预览读取失败'; };
  document.head.appendChild(loaded);
}
picker.addEventListener('change',update); update();
</script></html>"""
    (root / "trajectories.html").write_text(page.replace("OPTIONS", choices), encoding="utf-8")


def export_overview(root, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (time_ax, score_ax) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    methods = ("bezier", "hermite", "bspline")
    colors = ("#007789", "#a33e84", "#bd6512")
    for index, (method, color) in enumerate(zip(methods, colors)):
        rows = [row for row in paired if row["method"] == method]
        valid = [row for row in rows if row["baseline_accepted"] == row["toppquad_accepted"] == "True"]
        if valid:
            x = [float(row["delta_complete_s"]) for row in valid]
            y = [100 * float(row["delta_position_rmse_m"]) for row in valid]
            time_ax.scatter(x, y, alpha=.55, s=20, color=color, label=method)
        for arm, offset, opacity in (("baseline", -.17, .5), ("toppquad", .17, 1.)):
            count = sum(row[f"{arm}_accepted"] == "True" for row in rows)
            score_ax.bar(index + offset, 100 * count / len(rows), width=.30, color=color, alpha=opacity,
                         label=arm if index == 0 else None)
    for axis in (time_ax, score_ax):
        axis.grid(alpha=.22)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
    time_ax.axhline(0, color="#667085", linewidth=.8)
    time_ax.axvline(0, color="#667085", linewidth=.8)
    time_ax.set(title="Both arms accepted", xlabel="Completion time change (s)", ylabel="Position RMSE change (cm)")
    score_ax.set(title="All assigned pairs", xticks=range(3), xticklabels=methods, ylabel="Accepted (%)", ylim=(0, 105))
    fig.savefig(root / "overview.png", dpi=175)
    fig.savefig(root / "overview.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    root = parser.parse_args().report.resolve()
    design = load(root / "design.json")
    runs = load(root / "runs.csv")
    by_directory = {row["directory"]: row for row in runs}
    expected = {f'{task["id"]}/{method}/{arm}' for task in design["tasks"] for method in design["methods"]
                for arm in design["arms"]}
    assert len(by_directory) == len(runs) == len(expected) and by_directory.keys() == expected
    paired = load(root / "paired.csv")
    assert len(paired) * 2 == len(runs)
    audit = dict(planned_runs=len(expected), samples_verified=0, analytic_references=0,
                 paired_geometries_verified=0, planning_or_simulation_errors=[])
    pairs = []
    waypoint_rows = []
    failures = Counter()
    solver_rows = []
    for task in design["tasks"]:
        points = np.asarray(task["task"]["points_ned"])[:, [1, 0, 2]] * [1, 1, -1]
        for method in design["methods"]:
            key = f'{task["id"]}/{method}'
            pair = dict(points=points.tolist(), arms={})
            references = {}
            for arm in design["arms"]:
                relative = f"{key}/{arm}"
                folder = root / relative
                row = load(folder / "result.json")
                assert row["directory"] == relative and str(row["accepted"]) == by_directory[relative]["accepted"]
                preview = None
                if (folder / "reference_100.csv").is_file():
                    ref = references[arm] = load(folder / "reference.json")
                    assert ref["points"] == task["task"]["points_ned"]
                    ref_samples = np.loadtxt(folder / "reference_100.csv", delimiter=",", skiprows=1)
                    assert ref_samples.shape == (design["reference_export_points"], 16)
                    analytic = FrozenReference(ref)
                    probes = [analytic.sample(time)["p"] for time in ref_samples[:, 0]]
                    np.testing.assert_allclose(ref_samples[:, 1:4], np.asarray(probes)[:, [1, 0, 2]] * [1, 1, -1], atol=1e-7)
                    audit["analytic_references"] += 1
                    if "samples" in row:
                        preview = audit_samples(folder, row, design, ref)
                        audit["samples_verified"] += 1
                if "error" in row:
                    audit["planning_or_simulation_errors"].append(relative)
                if arm == "toppquad":
                    solver_log = folder / "solver.log"
                    log = solver_log.read_text(encoding="utf-8") if solver_log.is_file() else ""
                    raw_time = re.search(r"TOPPQuad Trajectory Time:\s*([0-9.eE+-]+)", log)
                    iterations = re.search(r"Iterations\s+(\d+)", log)
                    if raw_time and row.get("raw_toppquad_s") is not None:
                        assert abs(float(raw_time[1]) - row["raw_toppquad_s"]) < 1e-9
                    solver_rows.append(dict(route=task["id"], method=method,
                                            solver_returned_solution="Optimization Successful" in log,
                                            raw_reference_s=float(raw_time[1]) if raw_time else None,
                                            iterations=int(iterations[1]) if iterations else None,
                                            exported_reference_s=row.get("reference_s"),
                                            closed_loop_completed=row.get("sim_status") == "completed",
                                            accepted=row["accepted"], error=row.get("error")))
                if not row["accepted"]:
                    if row.get("error"):
                        failures[(method, arm, row["error"].split("; margins=", 1)[0])] += 1
                    elif row["plan_issues"]:
                        failures.update((method, arm, issue) for issue in row["plan_issues"])
                    else:
                        failures[(method, arm, "Closed-loop acceptance gate failed")] += 1
                waypoint_rows.extend(dict(route=task["id"], method=method, arm=arm, **evidence)
                                     for evidence in row.get("passages", ()))
                pair["arms"][arm] = dict(accepted=row["accepted"], complete_s=row.get("complete_s"), curves=preview)
            if len(references) == 2:
                audit_reference_matches(references["baseline"], references["toppquad"])
                audit["paired_geometries_verified"] += 1
            (root / key / "preview.js").write_text("window.rotorpyPair=" + json.dumps(pair, separators=(",", ":"))
                                                   + ";\n", encoding="utf-8")
            pairs.append(key)
    export_browser(root, pairs)
    export_overview(root, paired)
    task_points = [dict(route=task["id"], waypoint_index=index, north_m=north, east_m=east, up_m=-down,
                        heading_change_min_deg=task["heading_change_deg"][0],
                        heading_change_max_deg=task["heading_change_deg"][1])
                   for task in design["tasks"]
                   for index, (north, east, down) in enumerate(task["task"]["points_ned"])]
    write_csv(root / "task_points.csv", tuple(task_points[0]), task_points)
    write_csv(root / "solver.csv", tuple(solver_rows[0]), solver_rows)
    write_csv(root / "waypoints.csv",
              ("route", "method", "arm", "waypoint_index", "passed", "reference_time_s", "actual_time_s",
               "distance_m", "actual_speed_mps", "nearest_distance_m", "nearest_speed_mps"), waypoint_rows)
    write_csv(root / "failure_modes.csv", ("method", "arm", "reason", "count"),
              [dict(method=method, arm=arm, reason=reason, count=count)
               for (method, arm, reason), count in sorted(failures.items())])
    summary = load(root / "summary.json")

    def formatted(value, scale=1):
        return "—" if value is None else f"{scale*value:+.3f}"

    rows = []
    for method, item in summary.items():
        baseline = f'{item["baseline_accepted"]}/{item["pairs"]}'
        retimed = f'{item["toppquad_accepted"]}/{item["pairs"]}'
        d_time = formatted(item["median_delta_complete_s_both_accepted"])
        d_rmse = formatted(item["median_delta_position_rmse_m_both_accepted"], 100)
        rows.append(f'<tr><th>{method}</th><td>{baseline}</td><td>{retimed}</td>'
                    f'<td>{item["both_accepted"]}</td><td>{d_time}</td><td>{d_rmse}</td></tr>')
    aggregate = ('<h2>配对汇总</h2><table><tr><th>几何基线</th><th>原基线通过</th><th>TOPPQuad 通过</th>'
                 '<th>成对通过数</th><th>完成时间差中位数 (s)</th><th>位置 RMSE 差中位数 (cm)</th></tr>'
                 + "".join(rows) + '</table><p>失败原因表按出现次数统计。同一次可有多项未通过。</p>')
    index = root / "index.html"
    page = index.read_text(encoding="utf-8")
    panel = ('<p><a href="trajectories.html">逐轨迹三维查看</a> · <a href="audit.json">数据复核</a> · '
             '<a href="overview.svg">总览图</a> · <a href="task_points.csv">随机航点坐标</a> · '
             '<a href="waypoints.csv">逐航点误差</a> · <a href="solver.csv">离散求解数据</a> · '
             '<a href="failure_modes.csv">失败原因</a></p>')
    begin, end = "<!-- rotorpy-audit-start -->", "<!-- rotorpy-audit-end -->"
    section = panel + aggregate + "<h2>逐次数据</h2>"
    if begin in page:
        before, rest = page.split(begin, 1)
        page = before + begin + section + end + rest.split(end, 1)[1]
    else:
        page = page.replace("<table>", begin + section + end + "<table>", 1)
    index.write_text(page, encoding="utf-8")
    audit.update(result="PASS", result_files_sha256={file: digest(root / file)
                  for file in ("design.json", "runs.csv", "paired.csv", "summary.json", "task_points.csv", "waypoints.csv", "solver.csv",
                               "failure_modes.csv", "overview.svg")})
    write_json(root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

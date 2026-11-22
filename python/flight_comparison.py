"""Verify a selected AirSim batch against its recorded telemetry and render its two tables."""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np


def read_json(path):
    return path.read_text(encoding="utf-8-sig")


def flight_metrics(directory):
    report = json.loads(read_json(directory / "summary.json"))
    manifest = json.loads(read_json(directory / "live_manifest.json"))
    cleanup = json.loads(read_json(directory / "cleanup.json"))
    policy = report["config"]["preparation"]["task"]["policy"]
    geometry, arm = report["method"].rsplit("_", 1)
    row = dict(policy=policy, geometry=geometry, arm=arm, run=str(directory), outcome=report["outcome"],
               fingerprint=manifest["condition_fingerprint"], passed=False, error=report.get("error"),
               safe_landing=all(cleanup.get(k) for k in ("landed","disarmed","api_control_released")))
    settings = report["config"]["preparation"]["task"].get("retiming", {})
    if settings.get("frozen_source"):
        row["requested_fraction"] = settings["fractions"][0]
    with (directory / "states.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    row["samples"] = len(samples)
    if not samples or manifest.get("effective_duration_s") is None:
        return row
    def columns(*keys):
        return np.array([[float(sample[k]) for k in keys] for sample in samples])
    t = columns("t_s").ravel()
    p, v, q = columns("px","py","pz"), columns("vx","vy","vz"), columns("qx","qy","qz")
    T = float(manifest["effective_duration_s"])
    row.update(reference_s=T, execution_s=float(t[-1]), peak_speed_mps=float(np.linalg.norm(v,axis=1).max()),
               planning_s=(report.get("planning_evidence") or {}).get("parent_total_planning_wall_s"))
    if t[0] != 0 or np.any(np.diff(t)<=0) or t[-1]<T or len(t)<2:
        row["error"] = row["error"] or "执行样本未连续覆盖参考时段"
        return row
    plan = manifest["plan"]
    route = plan["route"]
    row["retiming_fraction"] = plan.get("retiming_fraction")
    speed_times = np.r_[t[t<T],T]
    row["reference_peak_speed_mps"] = float(np.linalg.norm(columns("qvx","qvy","qvz")[t<T],axis=1).max())
    actual_speed = np.array([np.interp(speed_times,t,axis) for axis in v.T]).T
    row["tracking_peak_speed_mps"] = float(np.linalg.norm(actual_speed,axis=1).max())
    if "waypoint_times" in plan:
        events = plan["waypoint_times"]
    else:
        from dual_target_control.route_curve import decode_route
        events = decode_route(plan).waypoint_times
    def at(data, time):
        return np.array([np.interp(time,t,axis) for axis in data.T])
    vias = []
    for i, event in enumerate(events):
        when = event["evaluation_time_s"]
        distance = float(np.linalg.norm(at(p, when)-event["point"]))
        speed = float(np.linalg.norm(at(v, when)))
        if i == len(events)-1:
            ok = distance<=.25 and speed<=.5
        elif policy == "stop":
            ok = distance<=route["waypoint_radius_m"] and speed<=route["stop_speed_limit_mps"]
        else:
            ok = distance<=route["waypoint_radius_m"] and speed>=route["minimum_pass_speed_mps"]
        reference_speed = event.get("reference_speed_mps")
        if reference_speed is None and arm == "ordinary":
            reference_speed = float(np.linalg.norm(route["waypoint_velocities"][i]))
        vias.append(dict(distance_m=distance, speed_mps=speed, reference_speed_mps=reference_speed, passed=bool(ok)))
    times = np.r_[t[t<T], T]
    errors = np.r_[np.linalg.norm((q-p)[t<T],axis=1), vias[-1]["distance_m"]]
    rmse = float(np.sqrt(np.trapz(errors**2,times)/T))
    max_error = float(errors.max())
    commands = columns("cmd_x","cmd_y","cmd_z")
    gaps = np.diff(t)
    cmd_peak = float(np.linalg.norm(commands,axis=1).max())
    slew_peak = float(np.max(np.linalg.norm(np.diff(commands,axis=0,prepend=np.zeros((1,3))),axis=1) /
                             np.maximum(np.r_[0.,gaps],.001)))
    recorded = report.get("metrics") or {}
    consistent = all(np.isclose(recorded.get(k,np.nan),value,rtol=1e-7,atol=1e-7) for k,value in (
        ("max_position_error_m",max_error), ("position_rmse_m",rmse),
        ("terminal_velocity_error_mps",vias[-1]["speed_mps"]), ("terminal_position_error_m",vias[-1]["distance_m"])))
    limits = route["command_limits"]
    peak_limit = route.get("peak_position_limit_m", .5)
    accepted = (row["outcome"]=="COMPLETED" and report.get("sources_unchanged") and row["safe_landing"] and
                row["samples"]==report["samples"] and consistent and all(w["passed"] for w in vias) and
                max_error<=peak_limit and gaps.max()<=.1+1e-9 and cmd_peak<=limits["max_speed_mps"]+1e-7 and
                slew_peak<=limits["max_slew_mps2"]+1e-7 and not any(s["paused"]=="True" for s in samples))
    failed = []
    if max_error>peak_limit:
        failed.append("全程位置")
    failed.extend(f"W{i+1}" for i,w in enumerate(vias[:-1]) if not w["passed"])
    if vias[-1]["distance_m"]>.25:
        failed.append("终点位置")
    if vias[-1]["speed_mps"]>.5:
        failed.append("终点速度")
    weights = np.maximum(0., np.minimum(t[1:],T)-t[:-1])
    slew_flags = [s["slew_clipped"]=="True" for s in samples[:-1]]
    speed_flags = [s["speed_clipped"]=="True" for s in samples[:-1]]
    row.update(speed_clipped_fraction=float(weights @ speed_flags / T),waypoints=vias, rmse_m=rmse, peak_error_m=max_error, max_command_mps=cmd_peak,
               max_command_slew_mps2=slew_peak, max_gap_s=float(gaps.max()),
               slew_clipped_fraction=float(weights@slew_flags/T), failed_checks=failed,
               agrees_with_summary=bool(consistent and recorded.get("task_success")==bool(accepted)),
               passed=bool(accepted and recorded.get("task_success")))
    return row


def render(directory):
    batch = json.loads(read_json(directory / "results.json"))
    rows = []
    for trial in batch["results"]:
        try:
            row = flight_metrics(Path(trial["run"]))
        except (KeyError, OSError, ValueError) as exc:
            geom, arm = trial["method"].rsplit("_",1)
            row = dict(policy=trial["policy"], geometry=geom, arm=arm, run=trial["run"],
                       error=str(exc), passed=False, samples=trial["samples"])
        rows.append(row)
    document = dict(requested=batch["expected"], recorded=len(rows), passed=sum(r["passed"] for r in rows), flights=rows)
    if batch.get("sweep_source"):
        document.update(sweep_source=batch["sweep_source"], requested_fractions=batch["requested_fractions"],
                        remaining_fractions=[job[3] for job in batch["remaining"]])
    (directory/"audit.json").write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>AirSim 轨迹对比</title><style>body{font:16px system-ui,sans-serif;color:#18332b;margin:36px auto;padding:0 22px;max-width:1250px}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #d9e2df;text-align:right}th{background:#eff5f2}td:first-child,th:first-child{text-align:left}a{color:#125d57}section{overflow:auto}small{color:#52665e}</style>',
            '<h1>AirSim 轨迹对比</h1><p>请求 %d 次，已有 %d 条记录，按原始采样复核通过 %d 次。</p>' %
            (document["requested"],document["recorded"],document["passed"])]
    for geometry, title in (("hermite","Hermite"),("bezier","Bézier")):
        group = [r for r in rows if r["geometry"]==geometry]
        if not group:
            continue
        html.extend([f"<h2>{title}</h2><section><table><thead><tr>"])
        for heading in ("任务 / 方法","计时比例","参考时长 / s","实测执行 / s","规划墙钟 / s",
                        "参考峰速 / m·s⁻¹","实测峰速 / m·s⁻¹","峰值误差 / m","终端速度 / m·s⁻¹","验收"):
            html.append(f"<th>{heading}</th>")
        html.append("</tr></thead><tbody>")
        for row in sorted(group,key=lambda r:(r["policy"],r["arm"])):
            target = quote(os.path.relpath(row["run"],directory).replace(os.sep,"/"),safe="/")
            policy = "连续过点" if row["policy"]=="pass" else "逐点停稳"
            method = "普通基线" if row["arm"]=="ordinary" else "TOPPQuad"
            html.append(f'<tr><td><a href="{target}/summary.json">{policy} / {method}</a></td>')
            terminal = (row.get("waypoints") or [{}])[-1].get("speed_mps")
            for value in [row.get(k) for k in ("retiming_fraction","reference_s","execution_s","planning_s","reference_peak_speed_mps","tracking_peak_speed_mps","peak_error_m")]+[terminal]:
                html.append("<td>—</td>" if value is None else f"<td>{value:.3f}</td>")
            detail = row.get("error") or "、".join(row.get("failed_checks", []))
            result = "通过" if row["passed"] else "未通过" + (f"<br><small>{escape(detail)}</small>" if detail else "")
            html.append(f"<td>{result}</td></tr>")
        html.append("</tbody></table></section>")
    html.append('<p><small>实测执行按 AirSim 时间戳从参考启动计至最后一帧。过点和终端指标插值到计划时刻。配对时应核对任务、条件指纹和完成状态。</small></p><a href="audit.json">逐航点实测与条件指纹</a></html>')
    (directory/"index.html").write_text("\n".join(html),encoding="utf-8")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch",type=Path)
    result = render(parser.parse_args().batch.resolve())
    print(f"{result['passed']} / {result['requested']}")

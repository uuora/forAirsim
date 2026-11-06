"""Explicit three-planner preparation and serial AirSim development coverage."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

from .planners import METHODS, LABELS, plan, common_fingerprint
from .runtime import ROOT, load_config, write_json, digest


TASK_CONFIG = ROOT / "configs/geometric_planner_task.json"
PLANNER_CONFIG = ROOT / "configs/geometric_planners_v1.json"


def prepare(task_path=TASK_CONFIG, output_path=PLANNER_CONFIG):
    task_path, output_path = Path(task_path), Path(output_path)
    spec = json.loads(task_path.read_text(encoding="utf-8"))
    return prepare_spec(spec,output_path,digest(task_path))


def prepare_spec(spec, output_path, spec_sha256):
    output_path=Path(output_path)
    cfg = json.loads((ROOT/spec["base_config"]).read_text(encoding="utf-8"))
    cfg.update(protocol=spec["protocol"], start=spec["task"]["points_ned"][0], yaw=spec["task"]["yaw"],
        threshold_status="development feasibility limits, frozen before flights; not a statistical or global-optimality experiment",
        comparison=dict(task=spec["task"], time_search=spec["time_search"], passage=spec["passage"],
                        task_spec_sha256=spec_sha256,
                        scope="three reference constructions, shared controller and local time search; real flight feasibility only"), cases={})
    if 'environment' in spec:
        cfg['comparison']['environment']=spec['environment']
    for method in METHODS:
        print("Planning "+method, flush=True)
        frozen = plan(method, spec["task"], spec["time_search"])
        cfg["cases"][method+"_pass"] = dict(label=LABELS[method]+" · 连续过点", method=method,
            points=spec["task"]["points_ned"][1:], segment_durations=frozen["segment_durations"],
            frozen_reference=frozen)
        print(f"  T={sum(frozen['segment_durations']):.6f}s, planning={frozen['planning']['wall_s']:.3f}s", flush=True)
    # Validate before replacing a previously generated plan.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        check=Path(temp)/'config.json';write_json(check,cfg);load_config(check)
    write_json(output_path, cfg)
    return output_path


def safe_cleanup(report):
    cleanup = report.get("cleanup", {})
    return all(cleanup.get(k) is True for k in ("landed", "disarmed", "api_control_released"))


def run_campaign(config_path=PLANNER_CONFIG, output=None, methods=METHODS, launch=False, runner=None):
    from .runtime import run, local_preflight
    cfg = load_config(config_path)
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Select one or more distinct supported planners")
    folder = Path(output or ROOT/"reports"/("geometric_three_planners_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"config_frozen.json", cfg)
    paths=list((ROOT/"python/geometric_validation").glob("*.py"))+[ROOT/"python/run_geometric_planner_comparison.py"]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    design = dict(config_sha256=digest(folder/"config_frozen.json"), sources=sources,
                  common_task_sha256=common_fingerprint(cfg), methods=list(methods),
                  local_preflight=local_preflight(cfg), no_automatic_retries=True)
    write_json(folder/"design_frozen.json", design)
    results, stopped = [], False
    for method in methods:
        if stopped:
            results.append(dict(method=method, status="NOT_RUN", reason="Previous flight did not confirm safe cleanup"))
        else:
            try:
                if any(digest(ROOT/p)!=v for p, v in sources.items()):
                    raise RuntimeError("Source changed during campaign")
                if load_config(config_path)!=cfg:
                    raise RuntimeError("Config changed during campaign")
                result = (runner or run)(config_path, method+"_pass", folder/"flights", launch=launch)
                results.append(dict(method=method, status="RECORDED", **result))
                stopped = not safe_cleanup(result["report"])
            except Exception as exc:
                results.append(dict(method=method, status="FAILED_BEFORE_RUN_RESULT", error=repr(exc)))
                stopped = True
        write_json(folder/"campaign.json", dict(design=design, results=results))
    rows = []
    for r in results:
        report = r.get("report", {}); metrics = report.get("metrics", {})
        rows.append(dict(method=r["method"], status=r["status"], normal_completion=report.get("normal_completion"),
            accepted=report.get("accepted"), tracking_passed=report.get("control_constraints_passed"),
            equivalent_class="bezier_hermite" if r['method'] in ('bezier','hermite') else 'bspline',
            passage_passed=report.get("waypoint_passage_passed"), audit_passed=report.get("data_audit_passed"),
            safe_cleanup=safe_cleanup(report), actual_time_s=report.get("actual_completion_time_s"),
            reference_time_s=sum(cfg["cases"][r["method"]+"_pass"]["segment_durations"]),
            planning_wall_s=report.get("planning_wall_s"),
            position_peak_m=metrics.get("position_error_m", {}).get("peak"),
            attitude_peak_deg=(metrics.get("attitude_reference_error_rad", {}).get("peak", 0)*180/3.141592653589793
                               if metrics.get("attitude_reference_error_rad") else None),
            run_dir=r.get("run_dir"), error=report.get("error", r.get("error"))))
    with (folder/"comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fingerprints=[]
    for result in results:
        if result.get('run_dir'):
            manifest=json.loads((Path(result['run_dir'])/'manifest.json').read_text(encoding='utf-8'))
            fingerprints.append(manifest.get('comparison_fingerprint'))
    comparable=bool(len(fingerprints)==len(methods) and None not in fingerprints and len(set(fingerprints))==1)
    write_json(folder/"summary.json", dict(results=rows, comparable_conditions=comparable,
        environment=cfg.get('comparison',{}).get('environment',dict(id='baseline',label='E0 · 基准三维过点')),
        scope="one attempt per planner, no reliability or superiority claim",
        equivalent_pair=["bezier", "hermite"], all_accepted=all(r.get("accepted") is True for r in rows)))
    if runner is None:
        from .comparison import render
        render(folder)
    return folder, rows

"""Bind completed scale audits and visibly retain crashed/interrupted attempts."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from analyze_t1_identity_flight import digest
from t1_scaled_motion_plan_v1 import validate_plan

ROOT=Path(__file__).resolve().parents[1]


def checked(path,expected=None):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/"logs") or not path.is_file():
        raise ValueError("PROJECT_LOG_EVIDENCE_REQUIRED")
    if expected is not None and digest(path)!=expected:
        raise ValueError("INPUT_EVIDENCE_CHANGED")
    return path


def interrupted_run(directory):
    directory=Path(directory).resolve();report_path=checked(directory/"flight/report.json")
    raw=json.loads(report_path.read_text(encoding="utf-8"));plan=raw["configuration"]
    if raw.get("finished_utc") or raw["status"]!="RUNNING":
        raise ValueError("INTERRUPTED_INITIAL_REPORT_REQUIRED")
    grid=validate_plan(plan)
    ledger=checked(directory/"flight/frame_results.jsonl")
    frames=[json.loads(s) for s in ledger.read_text(encoding="utf-8").splitlines() if s.strip()]
    events_path=checked(directory/"flight/execution.jsonl")
    events=[json.loads(s) for s in events_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    acquisition=checked(directory/"flight/acquisition_frame_results.jsonl")
    initial=[json.loads(s) for s in acquisition.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not frames or len({f["slot_index"] for f in frames})!=len(frames):
        raise ValueError("UNIQUE_PARTIAL_SAVED_FRAME_ROWS_REQUIRED")
    epoch=frames[0]["due_monotonic_s"]-frames[0]["slot_index"]*.4
    if any(not 0<=f["slot_index"]<grid["expected_main_slots"] or
           abs(f["due_monotonic_s"]-epoch-f["slot_index"]*.4)>1e-6 for f in frames):
        raise ValueError("PARTIAL_SAVED_GRID_CLOCK_MISMATCH")
    empty=dict(counts={},after_due_s=dict(count=0,minimum=None,median=None,maximum=None))
    r=dict(kind="T1_INTERRUPTED_SCALED_RUN_INTAKE",session_id=raw["session_id"],run=str(directory),
        architecture=plan["control_architecture"],mode=plan["mode"],
        new_flights=int(any(e.get("kind")=="SELF_STATE" and e.get("landed") is False for e in events)),
        input_receipts=[dict(path=str(p),sha256=digest(p)) for p in (directory/"manifest.json",report_path,ledger,events_path,acquisition)],
        plan_check=grid,scaled_motion_condition=plan["scaled_motion_condition"],
        base_evidence=dict(measured_epoch_monotonic_s=epoch,measured_frame_count=len(frames),acquisition_frame_count=len(initial),
            measured_identity_counts=dict(UNSCORED=len(frames)),
            independent_integration_qualification="UNVERIFIED_INTERRUPTED_NO_TERMINAL",
            fresh_ground_closure_verified=False,engine_closed_after_safe_completion=False,
            frame_rows=[dict(frame_id=f["frame_id"],readback_s=f["capture_received_host_s"],actual_phase=f["phase_id"],
                identity=dict(status="UNSCORED")) for f in frames]),
        endpoint=dict(status="UNVERIFIED_ENDPOINT",net_xy_displacement_m=None,start_state=None,end_state=None),
        endpoint_scale_status="UNVERIFIED_MISSING_TERMINAL_ENDPOINT",
        independent_scale_qualification="UNVERIFIED_INTERRUPTED_NO_TERMINAL",
        sampled_state_rows=[],frame_timing_rows=[],timing_summary=dict(frame_processing=empty,historical_full=empty),
        incomplete_frame_ledger=dict(path=str(ledger),sha256=digest(ledger),scope="saved pre-interruption frames, not independently scored"),
        incomplete_record_notice="中断流水未评分；缺少终态、端点与安全收尾证明",
        initial_report_preserved=True,formal_efficacy_results=False)
    return r


def collect(analyses,interrupted,output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/"logs"):
        raise ValueError("NEW_PROJECT_CAMPAIGN_OUTPUT_REQUIRED")
    runs=[];receipts=[]
    for path in analyses:
        path=checked(path);data=json.loads(path.read_text(encoding="utf-8"))
        if data["kind"]!="T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS" or data["formal_efficacy_results"]:
            raise ValueError("BOUNDED_SCALE_CHILD_ANALYSIS_REQUIRED")
        for row in data["runs"]:
            for record in row["input_receipts"]:checked(record["path"],record["sha256"])
            if row["base_evidence"]["measured_frame_count"]==0:
                row["incomplete_record_notice"]="UE启动阶段崩溃；无主帧，起飞状态与收尾未核实"
            runs.append(row)
        receipts.append(dict(path=str(path),sha256=digest(path)))
    if interrupted:runs.append(interrupted_run(interrupted))
    if len({r["session_id"] for r in runs})!=len(runs):
        raise ValueError("UNIQUE_ATTEMPT_SESSION_IDS_REQUIRED")
    order={"zero":0,"scale50":1,"scale100":2}
    runs.sort(key=lambda r:(order[r["scaled_motion_condition"]["name"]],r["session_id"]))
    result=dict(kind="T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS",created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,child_analysis_receipts=receipts,
        complete_scale_qualified_sessions=[r["session_id"] for r in runs if r["independent_scale_qualification"].startswith("PASS")],
        incomplete_sessions=[r["session_id"] for r in runs if not r["independent_scale_qualification"].startswith("PASS")],
        new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=["Startup crash and interrupted airborne attempt remain separate from qualified complete sessions.",
            "Partial frame ledgers are UNSCORED; missing terminal endpoint and safety proof remain unknown.",
            "Original full completion clocks, source budgets and control limits remain unchanged.",
            "Different predeclared durations qualify endpoint scale, not algorithm efficacy.",
            "Fixed-height XY development is not full geometric trajectory optimization."])
    output.mkdir(parents=True)
    report=output/"report.json";report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    copy=output/Path(__file__).name;shutil.copyfile(Path(__file__),copy)
    artifact=dict(kind="T1_SCALED_CAMPAIGN_BINDING_RECEIPT",created_utc=datetime.now(timezone.utc).isoformat(),
        artifacts=[dict(path=str(p),sha256=digest(p)) for p in (report,copy)],
        child_analysis_receipts=receipts,input_receipts=[x for r in runs for x in r["input_receipts"]],
        new_flights=0,formal_efficacy_results=False)
    (output/"artifact_receipt.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("bound campaign attempts",len(runs),"qualified",len(result["complete_scale_qualified_sessions"]),"incomplete",len(result["incomplete_sessions"]))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--analyses",nargs="+",type=Path,required=True)
    p.add_argument("--interrupted",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();collect(a.analyses,a.interrupted,a.output)

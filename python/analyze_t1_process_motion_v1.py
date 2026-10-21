"""Independently audit the declared larger scale, source chain and sampled stop."""
import argparse
from datetime import datetime,timezone
import json,shutil
from pathlib import Path

import analyze_t1_settlement as timing
from audit_t1_feedback_stop_v1p1 import feedback_audit
from analyze_t1_settlement_v1p1 import zero_hold_audit
from analyze_t1_settlement_v1p4 import admission_audit
from t1_pending_review_audit_v1 import pending_review_audit
from t1_process_plan_v1 import validate_plan
from audit_t1_process_control_v1 import audit as process_audit
from audit_t1_persistence_v1 import audit as persistence_audit
from audit_t1_feedback_dispatch_v1 import audit as dispatch_audit
from analyze_t1_identity_flight import digest


def process_replay_audit(root, report, events):
    issues=[]
    try:
        folder=Path(root)/'flight/control_process'
        original=[json.loads(s) for s in (folder/'execution.jsonl').read_text(encoding='utf-8').splitlines()]
        replayed=[e for e in events if e.get('process_trace_replay') is True]
        if len(original)!=len(replayed):raise ValueError('PROCESS_TRACE_REPLAY_DENOMINATOR_MISMATCH')
        for before,after in zip(original,replayed):
            for key,value in before.items():
                mapped={'journal_sequence':'child_journal_sequence',
                    'journal_enqueued_monotonic_s':'child_journal_enqueued_monotonic_s'}.get(key,key)
                if after.get(mapped)!=value:raise ValueError('PROCESS_TRACE_EXECUTION_CLOCK_OR_VALUE_CHANGED')
        final=json.loads((folder/'final.json').read_text(encoding='utf-8'))
        if report.get('control_process')!=final:raise ValueError('PROCESS_FINAL_REPORT_BINDING_CHANGED')
        if report.get('control_process_trace_replay',{}).get('events')!=len(original):
            raise ValueError('PROCESS_REPLAY_REPORT_COUNT_CHANGED')
    except (ValueError,KeyError,OSError,TypeError) as error:
        issues.append(str(error) if isinstance(error,ValueError) else type(error).__name__)
    return dict(status='PASS_PROCESS_TRACE_REPLAY' if not issues else 'FAIL_PROCESS_TRACE_REPLAY',issues=issues,
        execution_clock_rebased=False,post_control_replay=True,new_flights=0)


def scale_status(mode,net_m,minimum_mm,nonzero_commands):
    if mode=="ZERO_GATE":
        return "PASS_ZERO_GATE_DRIFT" if net_m is not None and net_m<=.005 and nonzero_commands==0 else "FAIL_ZERO_GATE"
    return "PASS_ENDPOINT_SCALED_MOTION" if net_m is not None and net_m>=minimum_mm/1000 else "FAIL_DECLARED_ENDPOINT_SCALE_NOT_REACHED"


def analyze_run(root):
    root=Path(root).resolve();raw=timing.read_json(root/"flight/report.json");plan=raw["configuration"]
    check=validate_plan(plan);r=timing.analyze_run(root);base=r["base_evidence"]
    events=[json.loads(s) for s in (root/"flight/execution.jsonl").read_text(encoding="utf-8").splitlines()]
    frames=raw["frames"];count=check["expected_main_slots"]
    grid=(len(frames)==len(raw["slots"])==count
        and {f["slot_index"] for f in frames}==set(range(count))
        and len({f["frame_id"] for f in frames})==count
        and all(s.get("status")=="CAPTURED" for s in raw["slots"])
        and all(f.get("evaluation_truth",{}).get("target",{}).get("visible_pixel_count",0)>=500 for f in frames))
    stop=feedback_audit(plan,raw,events);zero=zero_hold_audit(raw,events);admission=admission_audit(raw)
    pending=pending_review_audit(raw)
    persistence=persistence_audit(raw)
    dispatch=dispatch_audit(raw,events)
    process=process_audit(root/"flight/control_process")
    replay=process_replay_audit(root,raw,events)
    counts=base["measured_identity_counts"]
    no_wrong=not any(n for key,n in counts.items() if key not in ("CORRECT","NO_OUTPUT"))
    source_ok=(base["independent_integration_qualification"]=="PASS_CONTROLLED_DEVELOPMENT_SCOPE"
        and grid and no_wrong and stop["status"].startswith("PASS") and zero["status"].startswith("PASS")
        and admission["status"].startswith("PASS") and pending["status"].startswith("PASS") and persistence["status"].startswith("PASS") and dispatch["status"].startswith("PASS") and process["status"].startswith("PASS") and replay["status"].startswith("PASS"))
    qualified=scale_status(plan["mode"],r["endpoint"]["net_xy_displacement_m"],
        check["minimum_endpoint_mm"],base["nonzero_target_command_count"])
    r.update(plan_check=check,scaled_motion_condition=plan["scaled_motion_condition"],
        endpoint_scale_status=qualified,feedback_stop_audit=stop,zero_hold_audit=zero,
        actual_clear_admission_audit=admission,retained_pending_review_audit=pending,bounded_persistence_audit=persistence,feedback_dispatch_audit=dispatch,process_control_audit=process,process_trace_replay_audit=replay,
        measured_grid_check=dict(status="PASS_COMPLETE_VISIBLE_SCALED_GRID" if grid else "FAIL_MEASURED_SCALE_GRID",
            planned_slots=count,recorded_frames=len(frames)),
        independent_scale_qualification="PASS_BOUNDED_SCALED_MOTION_DEVELOPMENT" if source_ok and qualified.startswith("PASS") else
            "FAIL_DECLARED_SCALE" if source_ok else "FAIL_SCALED_MOTION_EVIDENCE",
        original_20mm_scale_status_retained=r["endpoint_scale_status"],
        formal_efficacy_results=False)
    return r


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("NEW_SCALE_ANALYSIS_REQUIRED")
    runs=[analyze_run(p) for p in roots];output.mkdir(parents=True)
    names=("analyze_t1_process_motion_v1.py","t1_process_plan_v1.py","audit_t1_process_control_v1.py","t1_process_trace_v1.py","t1_process_control_v1.py","analyze_t1_pipeline_motion_v1p2.py","t1_pipeline_plan_v1p2.py","analyze_t1_pipeline_motion.py","t1_pipeline_plan_v1.py","audit_t1_persistence_v1.py","audit_t1_feedback_dispatch_v1.py",
        "analyze_t1_scaled_motion.py","t1_scaled_motion_plan_v1.py","analyze_t1_settlement.py",
        "audit_t1_feedback_stop_v1p1.py","t1_stop_receipt_match.py","analyze_t1_settlement_v1p1.py",
        "analyze_t1_settlement_v1p4.py","t1_pending_review_audit_v1.py","analyze_t1_delayed_identity.py",
        "analyze_t1_identity_flight.py","analyze_t1_motion_scale.py","analyze_t1_async_timing.py")
    for name in names:
        dest=output/"source_snapshots"/name;dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name),dest)
    result=dict(kind="T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS",created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=["160/320-second conditions have new predeclared scale criteria.",
            "Original 20 mm trials and original 0.4-second full completion clocks remain.",
            "Fixed-height XY response is not full geometric trajectory optimization.",
            "Sampled stopping does not establish hard realtime or continuous physical safety."])
    timing.save(output/"report.json",result)
    timing.save(output/"artifact_receipt.json",dict(files=[dict(path=str(p),sha256=digest(p)) for p in output.rglob("*") if p.is_file()],
        input_receipts=[x for r in runs for x in r["input_receipts"]]))
    for r in runs:print(r["session_id"],r["independent_scale_qualification"],r["endpoint"]["net_xy_displacement_m"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();analyze(a.runs,a.output)

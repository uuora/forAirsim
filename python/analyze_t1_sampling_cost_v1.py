"""Offline producer checkpoint timings; preserve original slot and stop verdicts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    ("采集", "capture_started_host_s", "capture_received_host_s"),
    ("采集后至RGB保存开始", "capture_received_host_s", "rgb_persistence_started_monotonic_s"),
    ("RGB保存及校验", "rgb_persistence_started_monotonic_s", "rgb_persistence_completed_monotonic_s"),
    ("RGB保存后至观察器开始", "rgb_persistence_completed_monotonic_s", "observer_start"),
    ("观察器", "observer_start", "observer_end"),
    ("观察器后至表面处理结束", "observer_end", "surface_completed_monotonic_s"),
    ("表面处理后至提交控制", "surface_completed_monotonic_s", "enqueued_monotonic_s"),
    ("提交控制至决策可用", "enqueued_monotonic_s", "motion_decision_available_monotonic_s"),
    ("决策后至媒体处理结束", "motion_decision_available_monotonic_s", "frame_processing_completed_monotonic_s"),
]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def finite(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def delta(start,end):
    if not (finite(start) and finite(end)) or end < start:
        raise ValueError("INVALID_RECORDED_CHECKPOINT_ORDER")
    return end-start


def stats(values):
    values=sorted(v for v in values if finite(v))
    if not values:return {"count":0,"median_ms":None,"p95_nearest_rank_ms":None,"maximum_ms":None}
    return dict(count=len(values),median_ms=statistics.median(values)*1000,
                p95_nearest_rank_ms=values[math.ceil(.95*len(values))-1]*1000,maximum_ms=values[-1]*1000)


def frame_cost(frame):
    """The nine serial intervals telescope; the independent stop tail is separate."""
    clocks={key:frame.get(key) for _,a,b in CHECKPOINTS for key in (a,b)}
    clocks["observer_start"]=frame["observer_result"]["process_started_host_monotonic_s"]
    clocks["observer_end"]=frame["observer_result"]["process_completed_host_monotonic_s"]
    durations={label:delta(clocks[a],clocks[b]) for label,a,b in CHECKPOINTS}
    producer=delta(frame["capture_started_host_s"],frame["frame_processing_completed_monotonic_s"])
    if not math.isclose(sum(durations.values()),producer,abs_tol=1e-8):raise ValueError("SERIAL_INTERVALS_DO_NOT_TELESCOPE")
    due=frame["due_monotonic_s"];deadline=frame["deadline_monotonic_s"]
    start=frame["capture_started_host_s"];processed=frame["frame_processing_completed_monotonic_s"]
    full=frame["decision_completed_monotonic_s"]
    if full < processed:raise ValueError("FULL_COMPLETION_PRECEDES_PROCESSING")
    return dict(frame_id=frame["frame_id"],slot_index=frame["slot_index"],
        due_s=due,deadline_s=deadline,capture_started_s=start,processing_completed_s=processed,
        start_offset_s=start-due,producer_checkpoint_span_s=producer,
        processing_after_due_s=processed-due,original_full_after_due_s=full-due,
        independent_stop_tail_after_processing_s=full-processed,
        processing_late=processed>deadline,original_full_late=full>deadline,
        durations_s=durations)


def analyze(analysis,output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if not analysis.is_relative_to(ROOT) or not output.is_relative_to(ROOT/"logs"):raise ValueError("PROJECT_LOCAL_ANALYSIS_REQUIRED")
    if output.exists():raise FileExistsError("SAMPLING_COST_OUTPUT_EXISTS")
    source=read(analysis)
    if source.get("kind")!="T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS" or source.get("formal_efficacy_results") is not False:
        raise ValueError("SAVED_SCALE_DEVELOPMENT_ANALYSIS_REQUIRED")
    reports=[];excluded=[];all_rows=[];gaps=[];inputs=[dict(path=str(analysis),sha256=digest(analysis))]
    for run in source["runs"]:
        if run["endpoint"]["net_xy_displacement_m"] is None:
            excluded.append(dict(session_id=run["session_id"],reason="缺少终态；本次阶段分布不混入崩溃或未评分中断流水"));continue
        for item in run["input_receipts"]:
            if digest(item["path"])!=item["sha256"]:raise ValueError("SEALED_INPUT_CHANGED")
            inputs.append(item)
        raw=read(Path(run["run"])/"flight/report.json")
        rows=[frame_cost(f) for f in raw["frames"]]
        expected=run["base_evidence"]["measured_frame_count"]
        if len(rows)!=expected:raise ValueError("MEASURED_FRAME_COUNT_MISMATCH")
        frame_ids=[r["frame_id"] for r in rows]
        if len(set(frame_ids))!=len(rows):raise ValueError("DUPLICATE_FRAME_ID")
        late=sum(r["original_full_late"] for r in rows)
        processing_late=sum(r["processing_late"] for r in rows)
        if late!=run["timing_summary"]["historical_full"]["counts"].get("LATE",0):raise ValueError("ORIGINAL_FULL_TIMING_CHANGED")
        if processing_late!=run["timing_summary"]["frame_processing"]["counts"].get("LATE",0):raise ValueError("ORIGINAL_PROCESSING_TIMING_CHANGED")
        for row in rows:row["session_id"]=run["session_id"]
        stage_stats={label:stats([r["durations_s"][label] for r in rows]) for label,_,_ in CHECKPOINTS}
        stage_totals={label:sum(r["durations_s"][label] for r in rows) for label,_,_ in CHECKPOINTS}
        dominant=max(stage_totals,key=stage_totals.get)
        run_gaps=[]
        for slot in raw["slots"]:
            if slot.get("status")!="NOT_CAPTURED_SLOT_DEADLINE":continue
            preceding=[r for r in rows if r["slot_index"]<slot["slot_index"]]
            if not preceding:raise ValueError("MISSING_SLOT_WITHOUT_PRECEDING_FRAME")
            prev=max(preceding,key=lambda r:r["slot_index"])
            observed=slot["observed_monotonic_s"]
            unmeasured=delta(prev["processing_completed_s"],observed)
            gap=dict(session_id=run["session_id"],missing_slot=slot["slot_index"],status=slot["status"],
                due_s=slot["due_monotonic_s"],deadline_s=slot["deadline_monotonic_s"],check_observed_s=observed,
                previous_frame_id=prev["frame_id"],previous_slot=prev["slot_index"],
                previous_capture_start_offset_s=prev["start_offset_s"],previous_producer_span_s=prev["producer_checkpoint_span_s"],
                previous_processing_end_s=prev["processing_completed_s"],
                previous_processing_end_minus_missing_deadline_s=prev["processing_completed_s"]-slot["deadline_monotonic_s"],
                processing_end_to_slot_check_unattributed_s=unmeasured,
                previous_serial_durations_s=prev["durations_s"])
            run_gaps.append(gap)
        reports.append(dict(session_id=run["session_id"],measured_main_frames=len(rows),planned_main_slots=len(raw["slots"]),
            independent_scale_verdict_preserved=run["independent_scale_qualification"],original_full_late=late,processing_late=processing_late,
            stage_statistics=stage_stats,stage_total_seconds=stage_totals,largest_total_serial_stage=dominant,
            largest_stage_share_of_serial_span=stage_totals[dominant]/sum(stage_totals.values()),
            capture_start_offset=stats([r["start_offset_s"] for r in rows]),
            producer_checkpoint_span=stats([r["producer_checkpoint_span_s"] for r in rows]),
            original_full_after_due=stats([r["original_full_after_due_s"] for r in rows]),
            stop_tail_after_processing=stats([r["independent_stop_tail_after_processing_s"] for r in rows]),
            missing_slots=len(run_gaps)))
        gaps.extend(run_gaps);all_rows.extend(rows)
    output.mkdir(parents=True)
    def write_csv(path,rows):
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            if not rows:return
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    serial=[]
    for row in all_rows:
        flat={k:v for k,v in row.items() if k!='durations_s'}
        flat.update({k+'_ms':v*1000 for k,v in row['durations_s'].items()});serial.append(flat)
    gap_rows=[]
    for row in gaps:
        flat={k:v for k,v in row.items() if k!='previous_serial_durations_s'}
        flat.update({'上一帧_'+k+'_ms':v*1000 for k,v in row['previous_serial_durations_s'].items()});gap_rows.append(flat)
    write_csv(output/'frame_checkpoint_costs.csv',serial);write_csv(output/'missing_slot_context.csv',gap_rows)
    report=dict(kind='T1_OFFLINE_SAMPLING_COST_CHECKPOINT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS_OFFLINE_CHECKPOINT_ACCOUNTING',input_receipts=list({r['path']:r for r in inputs}.values()),
        new_flights=0,new_model_calls=0,formal_efficacy_results=False,completed_main_frames=len(all_rows),
        runs=reports,excluded_attempts=excluded,missing_slot_context=gaps,
        scope_notes=['九个主线程检查点区间可首尾相接；独立控制停止尾段可能与后帧重叠，单列且不加入串行成本。',
            '观察器包括检测、源属性判断、语义轮询与关联；已有时钟不能单独报告YOLO推理成本。',
            '采集区间包含RPC与渲染等待；不是相机曝光时长。',
            '表面处理后的区间包含自状态取得、规划和控制提交；不是纯规划CPU时间。',
            '帧处理结束后至下一槽检查缺少直接细分时钟；包含事件刷新、流水写入和循环调度，原因未知。',
            '中位数和95分位数不能跨阶段相加；这些帧是会话内嵌套记录，不是独立重复。',
            '检查点仅定位已记录阶段，不能证明CPU、磁盘、线程调度或VLX某一因素的因果作用。'])
    save(output/'report.json',report)
    snapshot=output/Path(__file__).name;snapshot.write_bytes(Path(__file__).read_bytes())
    artifacts=[dict(path=str(p),sha256=digest(p)) for p in (output/'report.json',output/'frame_checkpoint_costs.csv',output/'missing_slot_context.csv',snapshot)]
    save(output/'artifact_receipt.json',dict(kind='T1_OFFLINE_SAMPLING_COST_RECEIPT',input_receipts=report['input_receipts'],artifacts=artifacts,new_flights=0,new_model_calls=0,formal_efficacy_results=False))
    print('PASS_OFFLINE_CHECKPOINT_ACCOUNTING',len(all_rows),'main frames',len(gaps),'missing slots')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.analysis,a.output)

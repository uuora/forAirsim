"""Read saved T1 analysis for Qt replay; never create a simulator/API client."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_KINDS = {
    "T1_SOURCE_FAULT_DEVELOPMENT_ANALYSIS",
    "T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS",
    "T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS",
}
CONDITIONS = {"baseline": "零输出门控", "delivery_delay": "延迟交付",
              "occlusion": "遮挡与返回", "glint": "小白斑干扰"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(value, root=ROOT, *, suffixes=(".json",)):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() not in suffixes:
        raise ValueError("文件必须属于本项目，且类型应与保存记录一致。")
    if not path.is_file():
        raise ValueError("记录所引用的文件不存在。")
    return path


def read_json(path, root=ROOT):
    path = local_path(path, root)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("分析文件超过512 MB，请使用会话汇总文件。")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def load_clear_supplement(source, root=ROOT):
    receipt = source.get("interrupted_target_clear_supplement")
    if not receipt:
        return None
    path = local_path(receipt["path"], root)
    if sha256(path) != receipt.get("sha256"):
        raise ValueError("首次清零补充记录与登记的校验值不一致。")
    result = read_json(path, root)
    if (result.get("kind") != "T1_INTERRUPTED_TARGET_CLEAR_SUPPLEMENT"
            or result.get("session_id") != source["session_id"]
            or result.get("supplemental_only") is not True
            or result.get("overall_experiment_qualification_changed") is not False
            or not result.get("input_receipts")):
        raise ValueError("首次清零补充记录缺少会话绑定或范围声明。")
    for record in result["input_receipts"]:
        input_path = local_path(record["path"], root, suffixes=(".json", ".jsonl"))
        if sha256(input_path) != record.get("sha256"):
            raise ValueError("首次清零补充核查的原始输入发生了变化。")
    report_path = (Path(source["run"]) / "flight/report.json").resolve()
    if not any(Path(r["path"]).resolve() == report_path for r in result["input_receipts"]):
        raise ValueError("首次清零补充核查未绑定当前飞行报告。")
    if sha256(path) != receipt["sha256"]:
        raise ValueError("首次清零补充记录在读取期间变化。")
    return result


def latest_analysis(root=ROOT):
    protocol = read_json(root / "configs/t1_completion_protocol_v1.json", root)
    for key in ("latest_scaled_motion_development", "latest_source_fault_development",
                "latest_strict_settlement_development"):
        block = protocol.get(key, {})
        value = block.get("analysis_path") or block.get("analysis")
        if value:
            path = local_path(value, root)
            expected = block.get("analysis_sha256")
            if not expected or sha256(path) != expected:
                raise ValueError("最新分析与任务登记的校验值不一致，请核对保存记录。")
            return path
    raise ValueError("尚未找到任务一分析记录，可手动打开已保存的分析报告。")


@dataclass
class RunView:
    session_id: str
    label: str
    directory: Path
    planned: int
    measured: int
    acquisition: int
    counts: dict
    net_mm: float | None
    scale_status: str
    evidence_status: str
    epoch: float | None
    state_points: list
    timing_points: list
    identity_points: list
    report_digest: str | None
    minimum_mm: float | None
    source: dict = field(repr=False)


@dataclass
class CampaignView:
    path: Path
    digest: str
    runs: list[RunView]


@dataclass
class FrameView:
    slot: int
    frame_id: str
    phase: str
    time_s: float
    identity: str
    slot_status: str
    image: Path | None
    image_digest: str | None
    candidates: list
    selected: list | None
    semantic_reason: str
    motion_reason: str
    actual_time: bool = True


def load_campaign(path, root=ROOT):
    path = local_path(path, root)
    original_hash = sha256(path)
    raw = read_json(path, root)
    if raw.get("kind") not in ANALYSIS_KINDS or raw.get("formal_efficacy_results") is not False:
        raise ValueError("请选择任务一的开发分析报告，正式实验记录需使用对应入口。")
    if not isinstance(raw.get("runs"), list) or not raw["runs"]:
        raise ValueError("分析报告没有会话记录。")
    runs = []
    for source in raw["runs"]:
        supplement = load_clear_supplement(source, root)
        if supplement:
            source["verified_clear_supplement"] = supplement
        base = source["base_evidence"]
        epoch = base.get("measured_epoch_monotonic_s")
        if not finite(epoch) and base.get("measured_frame_count",0) != 0:
            raise ValueError("会话缺少主阶段的实际时钟。")
        epoch = epoch if finite(epoch) else None
        directory = local_path(Path(source["run"]) / "flight/report.json", root).parents[1]
        endpoint = source["endpoint"]
        net = endpoint.get("net_xy_displacement_m")
        condition = source.get("source_perturbation", {}).get("condition")
        scale_decl = source.get("scaled_motion_condition", {})
        label = CONDITIONS.get(condition, "可见目标运动")
        if scale_decl.get("minimum_endpoint_mm"):
            label = f"{scale_decl['minimum_endpoint_mm']:g}毫米尺度"
        if source["mode"] == "ZERO_GATE":
            label = "零输出门控"
        if "_rep" in source["session_id"]:
            repeat = source["session_id"].split("_rep", 1)[1].split("_", 1)[0]
            label += f" · 独立重复{repeat}"
        if "_retry" in source["session_id"]:
            retry = source["session_id"].split("_retry", 1)[1].split("_", 1)[0]
            label += f" · 初始化重试{retry}"
        if "SPAWN_PROCESS_OWN_STATE" in source.get("architecture", ""):
            label += " · V11控制进程"
        elif "BOUNDED_PERSISTENCE_V10P2" in source.get("architecture", ""):
            label += " · V10P2"
        elif "BOUNDED_PERSISTENCE_V10P1" in source.get("architecture", ""):
            label += " · V10P1"
        elif "BOUNDED_PERSISTENCE_V10" in source.get("architecture", ""):
            label += " · V10"
        label += " · " + source["session_id"].rsplit("_", 1)[-1]
        identities = [dict(frame_id=f["frame_id"],time_s=f["readback_s"]-epoch,
                           status=f["identity"]["status"])
                      for f in base["frame_rows"]
                      if epoch is not None and f["actual_phase"] != "initial_acquisition" and finite(f.get("readback_s"))]
        start = (endpoint.get("start_state") or {}).get("position_world_ned_m")
        states = []
        if epoch is not None and isinstance(start, list) and len(start) == 3:
            for state in source.get("sampled_state_rows", []):
                pos = state.get("position_world_ned_m")
                t = state.get("host_receipt_monotonic_s")
                if isinstance(pos, list) and len(pos)==3 and all(finite(x) for x in pos) and finite(t):
                    dx, dy = (pos[0]-start[0])*1000, (pos[1]-start[1])*1000
                    states.append(dict(time_s=t-epoch,x_mm=dx,y_mm=dy,net_mm=math.hypot(dx,dy)))
        receipts = [x for x in source.get("input_receipts", [])
                    if Path(x["path"]).resolve() == directory / "flight/report.json"]
        minimum = scale_decl.get("minimum_endpoint_mm", None if source["mode"]=="ZERO_GATE" else 20.)
        runs.append(RunView(
            session_id=source["session_id"],label=label,directory=directory,
            planned=source["plan_check"].get("expected_main_slots",base["measured_frame_count"]),
            measured=base["measured_frame_count"],acquisition=base.get("acquisition_frame_count",0),
            counts=dict(base["measured_identity_counts"]),net_mm=net*1000 if finite(net) else None,
            scale_status=source["endpoint_scale_status"],
            evidence_status=source.get("independent_fault_qualification",
                source.get("independent_scale_qualification",base["independent_integration_qualification"])),
            epoch=epoch,state_points=states,timing_points=source.get("frame_timing_rows",[]),
            identity_points=identities,report_digest=receipts[0]["sha256"] if len(receipts)==1 else None,
            minimum_mm=minimum,source=source))
    if sha256(path) != original_hash:
        raise ValueError("分析记录在读取期间发生了变化，请完成保存后重新打开。")
    return CampaignView(path,original_hash,runs)


def load_frames(run, root=ROOT):
    report_path = local_path(run.directory / "flight/report.json", root)
    if not run.report_digest or sha256(report_path) != run.report_digest:
        raise ValueError("会话报告与分析所记录的校验值不一致，停止显示配对结果。")
    raw = read_json(report_path, root)
    if raw.get("session_id") != run.session_id:
        raise ValueError("会话报告与分析的编号不一致。")
    scoring = {f["frame_id"]:f["identity"]["status"] for f in run.source["base_evidence"]["frame_rows"]}
    ledger=run.source.get("incomplete_frame_ledger")
    if ledger:
        ledger_path=local_path(ledger["path"],root,suffixes=(".jsonl",))
        if sha256(ledger_path)!=ledger["sha256"]:
            raise ValueError("中断流水与登记的校验值不一致。")
        with ledger_path.open(encoding="utf-8") as stream:
            actual_frames=[json.loads(line) for line in stream if line.strip()]
        if sha256(ledger_path)!=ledger["sha256"] or len(actual_frames)!=run.measured:
            raise ValueError("中断流水在读取期间变化或数量不一致。")
    else:actual_frames=raw["frames"]
    actual = {f["slot_index"]:f for f in actual_frames}
    frames = []
    for slot in raw["slots"]:
        i = slot["slot_index"]
        f = actual.get(i, {})
        media = f.get("media", {}).get("rgb", {})
        image = local_path(media["path"],root,suffixes=(".png",".jpg",".jpeg",".bmp")) if media.get("path") else None
        obs = f.get("observer_result", {})
        semantic = obs.get("semantic_frame", {})
        selected = obs.get("temporal_identity", {}).get("selected_candidate")
        selection = selected.get("bbox_xyxy_pixels") if isinstance(selected,dict) else None
        fid = f.get("frame_id", f"slot-{i:04d}")
        receipt = f.get("capture_received_host_s")
        actual_time=finite(receipt) and run.epoch is not None
        relative=receipt-run.epoch if actual_time else slot["offset_s"]
        frames.append(FrameView(i,fid,slot["phase_id"],relative,
            scoring.get(fid,"MISSING"),slot.get("status","UNKNOWN"),image,media.get("sha256"),
            [c["bbox_xyxy_pixels"] for c in semantic.get("candidates",[]) if "bbox_xyxy_pixels" in c],
            selection,obs.get("semantic_poll",{}).get("reason",""),
            f.get("motion_gate",{}).get("reason",""),actual_time))
    if sha256(report_path) != run.report_digest:
        raise ValueError("会话报告在读取期间发生了变化。")
    return frames


def validate_frame_image(frame):
    if frame.image is None:
        raise ValueError("此计划槽位没有保存图像。")
    if not frame.image_digest or sha256(frame.image) != frame.image_digest:
        raise ValueError("图像校验值与会话不一致，停止显示旧候选框。")
    return frame.image

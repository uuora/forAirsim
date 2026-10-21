"""Redraw saved task-one analysis in Chinese; never run or alter an experiment.

Usage: python plot_t1_identity_zh.py --analysis report.json --output NEW_DIRECTORY
The report schema is the delayed-chain analysis schema, also used by V5.
Only the supplied report is read as experimental input. A fresh output directory
is required; plots, their data mapping, source snapshot and hash receipt are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text


IDENTITY_STYLES = {
    "CORRECT": (2., "正确目标", "#247956", "o"),
    "NO_OUTPUT": (1., "无目标输出", "#657582", "o"),
    "AMBIGUOUS_OR_UNMATCHED": (.5, "含糊或未匹配", "#ba852c", "D"),
    "WRONG_DISTRACTOR": (0., "错选干扰物", "#b64240", "X"),
    "UNKNOWN": (-.5, "身份未知", "#84778c", "s"),
}
QUALIFICATION_ZH = {
    "PASS_CONTROLLED_DEVELOPMENT_SCOPE": "受控开发范围通过",
    "FAIL_WRONG_IDENTITY": "失败：错选目标",
    "FAIL_EVIDENCE_ISSUES": "失败：证据检查有问题",
    "FAIL_NO_VALID_MEASURED_OUTPUT": "失败：测量阶段无有效输出",
    "FAIL_NO_NONZERO_TARGET_COMMAND": "失败：无非零目标指令",
    "FAIL_RUNNER_STATUS": "失败：原运行器未通过",
    "FAIL_HANDOFF_EVIDENCE": "失败：交接证据有问题",
    "FAIL_ACTIVE_STALL_EVIDENCE": "失败：活动指令停顿证据不足",
}
BACKEND_ZH = {
    "REPLY_RETURNED": "返回",
    "READ_TIMEOUT": "读取超时",
    "CONNECT_TIMEOUT": "连接超时",
    "BACKEND_ERROR": "云端调用错误",
    "PENDING": "未记录完成",
    "UNKNOWN": "状态未知",
}
MEASURED_COLOR = "#3778a3"
RETURN_COLOR = "#52764d"
HIDDEN_COLOR = "#ad7359"
ATTRIBUTE_COLOR = "#c38822"
HANDOFF_COLOR = "#705695"
ACTIVE_BLOCK_COLOR = "#7c73a0"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def title_for_run(run: dict, index: int) -> str:
    sid = str(run.get("session_id", ""))
    version = re.search(r"(?:^|_)(v\d+(?:p\d+)?)(?:_|$)", sid, re.I)
    version_label = "（" + version.group(1).upper() + "）" if version else ""
    review = run.get("review_handoff", {}).get("configured") is True
    active = run.get("active_stall_qualification", {}).get("status", "NOT_DECLARED") != "NOT_DECLARED"
    if sid == "t1_identity_handoff_active_v5p1_20261005":
        name = "恢复调度修复与活动指令停顿"
    elif review and active:
        name = "后台交接与活动指令停顿"
    elif review and run.get("mode") == "ZERO_GATE":
        name = "后台复核交接·零速度门控"
    elif review and run.get("mode") == "SHORT_XY":
        name = "后台复核交接·短程 XY"
    elif run.get("mode") == "ZERO_GATE":
        name = "零速度联调"
    elif "attribute_delay" in sid:
        name = "源属性拒识与主线程停顿"
    elif run.get("mode") == "SHORT_XY":
        name = "短程运动联调"
    else:
        name = "保存试次 " + str(index + 1)
    return name + version_label


def qualification_zh(value) -> str:
    return QUALIFICATION_ZH.get(value, "其他资格状态（见数据映射）")


def runner_zh(value) -> str:
    return {"PASS": "通过", "FAIL": "失败"}.get(value, "其他状态（见数据映射）")


def identity_styles(runs: list[dict]) -> dict:
    present = {row["identity"]["status"] for run in runs for row in run["frame_rows"]}
    styles = {key: value for key, value in IDENTITY_STYLES.items() if key in present}
    for index, key in enumerate(sorted(present - set(IDENTITY_STYLES))):
        styles[key] = (-1. - index * .5, "其他身份状态 " + str(index + 1), "#84778c", "s")
    return styles


def origin_for(run: dict) -> float:
    value = run.get("acquisition_epoch_monotonic_s")
    return float(value if finite(value) else min(row["readback_s"] for row in run["frame_rows"]))


def phase_markers(run: dict) -> dict:
    result = {}
    for transition in run.get("phase_transitions", []):
        phase = transition.get("phase_id")
        value = transition.get("completed_monotonic_s")
        if phase in ("hidden", "return") and finite(value):
            result[phase] = value
    return result


def missing_slots(run: dict) -> list[dict]:
    rows = ((run.get("timing") or {}).get("frame_timing") or {}).get("rows", [])
    return [row for row in rows if row.get("slot_status") != "CAPTURED" and not row.get("frame_id")]


def measured_rows(run: dict) -> list[dict]:
    return [row for row in run["frame_rows"] if row.get("actual_phase") != "initial_acquisition"]


def committed_readbacks(run: dict) -> list[dict]:
    """Recorded output-frame readbacks, never a fabricated adoption clock."""
    return [row for row in run.get("review_handoff", {}).get("frame_rows", [])
            if row.get("review_handoff", {}).get("event") == "REVIEW_COMMITTED"
            and finite(row.get("readback_s"))]


def actual_active_block(run: dict) -> dict | None:
    qualification = run.get("active_stall_qualification", {})
    block = qualification.get("block", {})
    start, end = block.get("started_monotonic_s"), block.get("completed_monotonic_s")
    if qualification.get("status", "NOT_DECLARED") != "NOT_DECLARED" and finite(start) and finite(end) and end > start:
        return block
    return None


def transport_label(trace: dict) -> tuple[str, str, str]:
    status = trace.get("backend_status", "PENDING")
    error = trace.get("transport_error", trace.get("error_category"))
    if status == "REPLY_RETURNED":
        attribute = trace.get("source_attribute_guard_status", "NOT_CONFIGURED")
        if attribute == "REJECTED":
            return "返回；源属性拒绝", ATTRIBUTE_COLOR, "X"
        if attribute == "ACCEPTED":
            return "返回；属性检查通过", "#247956", "o"
        return "返回", "#247956", "o"
    label = BACKEND_ZH.get(error or status, "其他云端状态（见记录）")
    return label, "#b64240" if finite(trace.get("backend_completed_monotonic_s")) else "#657582", "x"


def validate_report(report: dict) -> list[dict]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("输入分析报告必须包含 runs 数组")
    usable = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("试次记录格式无效")
        rows = run.get("frame_rows", [])
        if not rows:
            continue
        for row in rows:
            if not finite(row.get("readback_s")) or not isinstance(row.get("identity", {}).get("status"), str):
                raise ValueError("保存帧缺少有效时间或身份状态；不能静默丢弃")
        usable.append(run)
    if not usable:
        raise ValueError("输入报告没有可绘制的保存帧")
    return usable


def decorate_phase(ax, run: dict, origin: float) -> None:
    measured = run.get("measured_epoch_monotonic_s")
    if finite(measured):
        ax.axvline(measured - origin, color=MEASURED_COLOR, ls="--", lw=1.3, zorder=1)
    phase = phase_markers(run)
    if "hidden" in phase and "return" in phase:
        ax.axvspan(phase["hidden"] - origin, phase["return"] - origin,
                   color=HIDDEN_COLOR, alpha=.16, zorder=0)
    if "return" in phase:
        ax.axvline(phase["return"] - origin, color=RETURN_COLOR, ls=":", lw=1.6, zorder=1)
    block = actual_active_block(run)
    if block:
        ax.axvspan(block["started_monotonic_s"] - origin, block["completed_monotonic_s"] - origin,
                   color=ACTIVE_BLOCK_COLOR, alpha=.16, zorder=0)


def timeline_figure(runs: list[dict], styles: dict):
    single = len(runs) == 1
    fig, axes = plt.subplots(len(runs), 2, figsize=(17, max(7.2, 3.4 * len(runs) + 1.45)), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, top=.85 if single else .91,
                        bottom=.27 if single else .16, hspace=.61, wspace=.22)
    fig.suptitle("既有联调数据：逐帧身份输出与语义请求时序", fontsize=17, y=.982)
    all_missing = any(missing_slots(run) for run in runs)
    has_commits = any(committed_readbacks(run) for run in runs)
    has_active_block = any(actual_active_block(run) for run in runs)
    if (has_commits or has_active_block) and not single:
        fig.subplots_adjust(bottom=.175)
    missing_level = min(value[0] for value in styles.values()) - .6
    metadata = []
    for index, run in enumerate(runs):
        left, right = axes[index]
        origin = origin_for(run)
        rows = run["frame_rows"]
        for key, (level, label, color, marker) in styles.items():
            subset = [row for row in rows if row["identity"]["status"] == key]
            if subset:
                left.scatter([row["readback_s"] - origin for row in subset], [level] * len(subset),
                             s=31, c=color, marker=marker, zorder=4, edgecolors="white", linewidths=.4)
        missing = missing_slots(run)
        visible_missing = [row for row in missing if finite(row.get("nominal_deadline_s"))]
        if visible_missing:
            left.scatter([row["nominal_deadline_s"] - origin for row in visible_missing],
                         [missing_level] * len(visible_missing), color="#534c4a", marker="|", s=110, zorder=4)
        scored = {row["frame_id"]: row for row in rows}
        commits = committed_readbacks(run)
        drawn_commits = []
        for row in commits:
            target = scored.get(row.get("frame_id"))
            if target is not None:
                level = styles[target["identity"]["status"]][0]
                left.scatter([row["readback_s"] - origin], [level], marker="D", s=85,
                             facecolors="none", edgecolors=HANDOFF_COLOR, linewidths=1.2, zorder=5)
                drawn_commits.append(dict(frame_id=row["frame_id"], readback_s=row["readback_s"],
                    event="REVIEW_COMMITTED", actual_adoption_time_inferred=False))
        levels = sorted((value[0], value[1]) for value in styles.values())
        if all_missing:
            levels.insert(0, (missing_level, "未采集槽"))
        left.set_yticks([row[0] for row in levels], [row[1] for row in levels])
        left.set_ylim(min(row[0] for row in levels) - .35, max(row[0] for row in levels) + .35)
        left.set_ylabel("独立身份判定")
        name = title_for_run(run, index)
        left.set_title(f"（{2 * index + 1}）{name}\n原运行器：{runner_zh(run.get('original_runner_status'))}；"
                       f"独立资格：{qualification_zh(run.get('independent_integration_qualification'))}",
                       fontsize=11, pad=10)
        right.set_title(f"（{2 * index + 2}）语义请求的云端耗时与返回状态", fontsize=11, pad=10)
        request_metadata = []
        traces = run.get("sanitized_transport_trace", [])
        extent = max(row["readback_s"] - origin for row in rows)
        for request_index, trace in enumerate(traces):
            start = trace.get("backend_started_monotonic_s")
            end = trace.get("backend_completed_monotonic_s")
            label, color, marker = transport_label(trace)
            if finite(start):
                x0 = start - origin
                if finite(end):
                    x1 = end - origin
                    right.plot([x0, x1], [request_index + 1] * 2, color=color, lw=3.5, solid_capstyle="butt")
                    right.scatter([x1], [request_index + 1], marker=marker, color=color, s=36, zorder=3)
                    right.text(x1 + .17, request_index + 1, label, fontsize=9, va="center")
                    extent = max(extent, x1)
                else:
                    right.scatter([x0], [request_index + 1], marker="x", color=color, s=32)
                    right.text(x0 + .17, request_index + 1, label, fontsize=9, va="center")
                    extent = max(extent, x0)
            request_metadata.append(dict(request_number=request_index + 1,
                request_id=trace.get("request_binding", {}).get("request_id"),
                backend_status=trace.get("backend_status"), source_attribute_status=trace.get("source_attribute_guard_status"),
                start_s_from_acquisition=start-origin if finite(start) else None,
                completion_s_from_acquisition=end-origin if finite(end) else None,
                chinese_label=label, evidence_adoption_inferred=False))
        if visible_missing:
            extent = max(extent, max(row["nominal_deadline_s"] - origin for row in visible_missing))
        for ax in (left, right):
            decorate_phase(ax, run, origin)
            ax.set_xlabel("距初始身份获取起点的时间（秒）")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#d9dee2", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
        left.set_xlim(-.15, extent + .35)
        right.set_xlim(-.15, extent + max(3.6, extent * .20))
        for ax in (left, right):
            # Auto locators also retain ticks outside the view interval. Use
            # only displayed ticks so the text-bound receipt audits real labels.
            ax.set_xticks([tick for tick in ax.get_xticks() if 0 <= tick <= ax.get_xlim()[1]])
        right.set_ylim(.4, max(1, len(traces)) + .6)
        right.set_yticks(range(1, len(traces) + 1), ["请求 " + str(i + 1) for i in range(len(traces))])
        right.set_ylabel("语义请求编号")
        metadata.append(dict(session_id=run.get("session_id"), human_label=name,
            original_runner_status=run.get("original_runner_status"),
            independent_integration_qualification=run.get("independent_integration_qualification"),
            acquisition_origin_monotonic_s=origin, saved_frame_count=len(rows),
            identity_counts=dict(Counter(row["identity"]["status"] for row in rows)),
            missing_measured_slots=missing, displayed_missing_slot_count=len(visible_missing),
            omitted_missing_time_count=len(missing)-len(visible_missing), requests=request_metadata))
        metadata[-1].update(review_committed_output_readbacks=drawn_commits,
            committed_receipts_without_saved_identity_frame=len(commits)-len(drawn_commits),
            actual_active_block=actual_active_block(run),
            active_stall_qualification=run.get("active_stall_qualification"))
    identity_handles = [Line2D([], [], marker=style[3], color="none", markerfacecolor=style[2],
                               markeredgecolor=style[2], label=style[1], markersize=7) for style in styles.values()]
    if all_missing:
        identity_handles.append(Line2D([], [], marker="|", color="none", markeredgecolor="#534c4a",
                                       label="未采集槽（在槽截止时刻标记）", markersize=11))
    identity_legend_y = .14 if single else .105 if has_commits or has_active_block else .083
    fig.legend(handles=identity_handles, loc="lower center", bbox_to_anchor=(.5, identity_legend_y),
               ncol=min(5, len(identity_handles)), frameon=False, fontsize=10)
    phase_handles = [Line2D([], [], color=MEASURED_COLOR, ls="--", label="测量起点"),
        Patch(color=HIDDEN_COLOR, alpha=.16, label="实际遮挡区间"),
        Line2D([], [], color=RETURN_COLOR, ls=":", label="目标实际返回"),
        Line2D([], [], marker="X", color=ATTRIBUTE_COLOR, label="返回后源属性拒绝", markersize=7)]
    if has_commits:
        phase_handles.append(Line2D([], [], marker="D", color="none", markeredgecolor=HANDOFF_COLOR,
                                   markerfacecolor="none", label="含交接记录的帧读回点", markersize=7))
    if has_active_block:
        phase_handles.append(Patch(color=ACTIVE_BLOCK_COLOR, alpha=.16, label="实际活动指令停顿区间"))
    fig.legend(handles=phase_handles, loc="lower center", bbox_to_anchor=(.5, .06 if single else .041), ncol=3 if len(phase_handles)>4 else 4,
               frameon=False, fontsize=10)
    note = "注：时间来自保存的主机时钟；云端返回或属性检查通过均不等于证据采用或身份保证。"
    if has_commits:
        note += "\n交接标记表示含 REVIEW_COMMITTED 记录的输出帧读回点，不代表精确采纳时刻。"
    fig.text(.5, .014 if single else .011 if has_commits else .018,
             note, ha="center", fontsize=9 if has_commits else 10, color="#46515a")
    return fig, metadata


def count_figure(runs: list[dict], styles: dict):
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, max(5.4, 1.4 * len(runs) + 1.7)),
                                    gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.19, right=.96, top=.83, bottom=.24, wspace=.23)
    fig.suptitle("既有联调数据：测量阶段输出与非零目标指令", fontsize=16, y=.96)
    metadata = []
    max_frames = max_commands = 0
    for index, run in enumerate(runs):
        rows = measured_rows(run)
        counts = Counter(row["identity"]["status"] for row in rows)
        planned = sum(run.get("measured_slot_status_counts", {}).values())
        timing_planned = ((run.get("timing") or {}).get("frame_timing") or {}).get("planned_slots")
        if isinstance(timing_planned, int) and not isinstance(timing_planned, bool):
            planned = timing_planned
        if planned < len(rows):
            raise ValueError("测量计划槽少于保存帧，不能绘制分母")
        missing_count = planned - len(rows)
        offset = 0
        for key, style in styles.items():
            count = counts[key]
            if count:
                left.barh(index, count, left=offset, color=style[2], edgecolor="white", height=.53)
                left.text(offset + count / 2, index, str(count), ha="center", va="center", color="white", fontsize=11)
                offset += count
        if missing_count:
            left.barh(index, missing_count, left=offset, facecolor="#efeeec", edgecolor="#534c4a", hatch="///", height=.53)
            left.text(offset + missing_count / 2, index, str(missing_count), ha="center", va="center", fontsize=10)
        left.text(0, index - .39, f"保存 {len(rows)} / 计划 {planned}；初始获取 {run.get('acquisition_frame_count', 0)} 帧", fontsize=9)
        scores = {row["frame_id"]: row["identity"]["status"] for row in run["frame_rows"]}
        command_counts = Counter()
        for command in run.get("target_command_rows", []):
            status = scores.get(command.get("target_frame_id"), "UNKNOWN")
            category = "CORRECT" if status == "CORRECT" else "WRONG_DISTRACTOR" if status == "WRONG_DISTRACTOR" else "UNKNOWN"
            command_counts[category] += 1
        declared = run.get("nonzero_target_command_count")
        if declared != sum(command_counts.values()):
            raise ValueError("非零目标指令总数与逐指令记录不一致")
        offset = 0
        for key, label, color in (("CORRECT", "正确目标指令", "#247956"),
                                  ("WRONG_DISTRACTOR", "错选目标指令", "#b64240"),
                                  ("UNKNOWN", "其他或未知身份指令", "#84778c")):
            count = command_counts[key]
            if count:
                right.barh(index, count, left=offset, color=color, edgecolor="white", height=.53)
                right.text(offset + count / 2, index, str(count), ha="center", va="center", color="white", fontsize=11)
                offset += count
        if not offset:
            right.text(.15, index, "0（零速度门控）" if run.get("mode") == "ZERO_GATE" else "0", va="center", fontsize=10)
        max_frames = max(max_frames, planned)
        max_commands = max(max_commands, offset)
        metadata.append(dict(session_id=run.get("session_id"), measured_saved_frames=len(rows),
                             measured_planned_slots=planned, measured_identity_counts=dict(counts),
                             measured_missing_slots=missing_count, nonzero_command_counts=dict(command_counts)))
    labels = [title_for_run(run, index) + "\n独立资格：" + qualification_zh(run.get("independent_integration_qualification"))
              for index, run in enumerate(runs)]
    left.set_yticks(range(len(runs)), labels, fontsize=9)
    right.set_yticks(range(len(runs)), [])
    for ax in (left, right):
        ax.set_ylim(len(runs) - .45, -.65)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d9dee2", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_xlim(0, max_frames + 1.5)
    right.set_xlim(0, max(4, max_commands + 2))
    left.set_title("实际保存输出与未采集槽", fontsize=12)
    right.set_title("按目标帧的独立身份判定分组", fontsize=12)
    left.set_xlabel("测量阶段槽数")
    right.set_xlabel("非零目标指令条数")
    handles = [Patch(color=style[2], label=style[1]) for style in styles.values()]
    if any(row["measured_missing_slots"] for row in metadata):
        handles.append(Patch(facecolor="#efeeec", edgecolor="#534c4a", hatch="///", label="未采集槽"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .10), ncol=min(5, len(handles)), frameon=False, fontsize=10)
    fig.text(.5, .055, "注：左图的无输出与未采集分别计数；右图只统计保存的非零目标指令。", ha="center", fontsize=10)
    fig.text(.5, .021, "这些试次属于开发联调；不同版本与场景的计数不能解释为正式对照收益或轨迹优化效果。", ha="center", fontsize=10, color="#46515a")
    return fig, metadata


def active_stall_figure(runs: list[dict]):
    """Display recorded dispatch/explicit-clear clocks and SELF_STATE receipts.

    Declared lease end is a declaration marker. The target setpoint interval
    ends only at the saved explicit zero dispatch, never at dispatch+duration.
    """
    usable, skipped = [], []
    for index, run in enumerate(runs):
        qualification = run.get("active_stall_qualification", {})
        if qualification.get("status", "NOT_DECLARED") == "NOT_DECLARED":
            continue
        block = actual_active_block(run)
        dispatch = qualification.get("dispatch_monotonic_s")
        clear = qualification.get("explicit_clear_monotonic_s")
        trigger = qualification.get("trigger_frame_id")
        command = next((row for row in run.get("target_command_rows", [])
                        if row.get("target_frame_id") == trigger and row.get("dispatch_monotonic_s") == dispatch), {})
        declared = command.get("lease_clear", {}).get("declared_lease_end_s")
        cadence = (run.get("controller_timing") or {}).get("monitor_event_receipt_cadence", {})
        receipts = cadence.get("receipts_monotonic_s", [])
        if (not block or not all(finite(value) for value in (dispatch, clear, declared))
                or clear < dispatch or not receipts):
            skipped.append(dict(session_id=run.get("session_id"), reason="MISSING_RECORDED_BLOCK_DISPATCH_CLEAR_DECLARATION_OR_QUERY_RECEIPTS"))
            continue
        if not all(finite(value) for value in receipts):
            raise ValueError("活动停顿的保存自身状态查询时刻无效；不能静默丢弃")
        usable.append((index, run, block, dispatch, clear, declared, cadence))
    if not usable:
        return None, dict(plotted_runs=[], not_plotted=skipped)
    fig, axes = plt.subplots(len(usable), 2, figsize=(15.5, 4.6 * len(usable) + 1.4), squeeze=False,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.14, right=.96, top=.85, bottom=.29, hspace=.55, wspace=.24)
    fig.suptitle("活动指令停顿：保存的租约声明、显式清零与自身查询", fontsize=16, y=.96)
    records = []
    for plot_index, (run_index, run, block, dispatch, clear, declared, cadence) in enumerate(usable):
        left, right = axes[plot_index]
        start, end = block["started_monotonic_s"], block["completed_monotonic_s"]
        x0 = min(start, dispatch, declared, clear) - .12
        x1 = max(end, dispatch, declared, clear) + .12
        receipts = [value for value in cadence["receipts_monotonic_s"] if x0 <= value <= x1]
        inside = [value for value in receipts if start <= value <= end]
        gaps = [row for row in cadence.get("gap_rows", [])
                if finite(row.get("left_receipt_s")) and finite(row.get("right_receipt_s"))
                and row["left_receipt_s"] >= x0 and row["right_receipt_s"] <= x1 and finite(row.get("gap_s"))]
        left.plot([0., end - start], [4., 4.], color=ACTIVE_BLOCK_COLOR, linewidth=8, solid_capstyle="butt")
        left.text((end - start) / 2, 4.22, f"实际阻塞 {end - start:.3f} 秒", ha="center", fontsize=10)
        left.plot([dispatch - start, clear - start], [3., 3.], color=MEASURED_COLOR, linewidth=6, solid_capstyle="butt")
        left.scatter([dispatch - start], [3.], marker="o", color=MEASURED_COLOR, s=40, zorder=4)
        left.scatter([declared - start], [2.], marker="|", color=ATTRIBUTE_COLOR, s=170, zorder=4)
        left.text(declared - start + .02, 2., f"{declared - start:+.3f} 秒", va="center", fontsize=10)
        left.scatter([clear - start], [1.], marker="s", color="#247956", s=44, zorder=4)
        left.text(clear - start + .02, 1., f"{clear - start:+.3f} 秒（显式派发）", va="center", fontsize=10)
        left.scatter([value - start for value in receipts], [0.] * len(receipts), marker="|", color="#46515a", s=100, zorder=4)
        left.set_yticks([0, 1, 2, 3, 4], ["自身状态查询读回", "显式清零派发", "声明租约终点", "非零目标设定至显式清零", "主线程实际阻塞"])
        left.set_ylim(-.45, 4.7)
        left.set_title(title_for_run(run, run_index), fontsize=12)
        if gaps:
            right.plot([row["right_receipt_s"] - start for row in gaps],
                       [row["gap_s"] * 1000 for row in gaps], "o-", color="#46515a", markersize=4, linewidth=.8)
        right.set_ylabel("相邻保存查询读回间隔（毫秒）")
        right.set_title(f"阻塞内 {len(inside)} 个实际查询读回点", fontsize=12)
        right.set_ylim(bottom=0)
        for ax in (left, right):
            ax.axvspan(0., end - start, color=ACTIVE_BLOCK_COLOR, alpha=.12, zorder=0)
            ax.axvline(declared - start, color=ATTRIBUTE_COLOR, ls="--", linewidth=1.)
            ax.axvline(clear - start, color="#247956", ls=":", linewidth=1.2)
            ax.set_xlim(x0 - start, x1 - start)
            ax.set_xlabel("距实际主线程阻塞起点的时间（秒）")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#d9dee2", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([tick for tick in ax.get_xticks() if x0 - start <= tick <= x1 - start])
        records.append(dict(session_id=run.get("session_id"), trigger_frame_id=qualification.get("trigger_frame_id"),
            active_stall_qualification=qualification.get("status"),
            independent_combined_development_qualification=run.get("independent_combined_development_qualification"),
            block_started_monotonic_s=start, block_completed_monotonic_s=end,
            actual_dispatch_monotonic_s=dispatch, explicit_zero_clear_monotonic_s=clear,
            declared_lease_end_monotonic_s=declared, displayed_self_state_receipts_monotonic_s=receipts,
            self_state_receipts_inside_actual_block=len(inside), displayed_query_gap_rows=gaps,
            command_lease_audit=command.get("lease_clear"), automatic_stop_at_declared_end_inferred=False,
            physical_stop_time_inferred=False))
    handles = [Patch(color=ACTIVE_BLOCK_COLOR, alpha=.16, label="实际主线程阻塞区间"),
        Line2D([], [], color=ATTRIBUTE_COLOR, ls="--", label="声明租约终点"),
        Line2D([], [], color="#247956", ls=":", label="保存的显式清零派发")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .12), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .064, "注：租约声明到期不表示自动停车；设定清除以保存的显式清零派发记录为准。", ha="center", fontsize=10)
    fig.text(.5, .026, "图示为主机派发与查询记录，未给出精确物理停车时刻，也不能证明硬实时或连续安全。", ha="center", fontsize=10, color="#46515a")
    return fig, dict(plotted_runs=records, not_plotted=skipped)


def figure_texts(figures: list) -> list[str]:
    # A draw first materializes numeric tick labels for the coverage audit.
    result = []
    for fig in figures:
        fig.canvas.draw()
        result += [text.get_text() for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    return result


def choose_font(required: set[int], override: Path | None) -> tuple[Path, dict]:
    candidates = [override] if override else [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/Deng.ttf"), Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]
    if not override:
        candidates += [Path(row.fname) for row in font_manager.fontManager.ttflist
                       if any(token in row.name for token in ("Noto Sans CJK", "Noto Sans SC", "Source Han Sans", "Microsoft YaHei", "SimHei"))]
    attempts = []
    for path in dict.fromkeys(candidates):
        if path is None or not path.is_file():
            continue
        font = FT2Font(str(path))
        missing = sorted(required - set(font.get_charmap()))
        attempts.append(dict(path=str(path.resolve()), family_name=font.family_name,
                             missing_codepoints=[f"U+{code:04X}" for code in missing]))
        if not missing:
            return path.resolve(), dict(path=str(path.resolve()), family_name=font.family_name,
                sha256=digest(path), required_codepoint_count=len(required),
                required_codepoints=[f"U+{code:04X}" for code in sorted(required)],
                missing_codepoints=[], glyph_coverage="PASS", candidate_checks=attempts)
    raise ValueError("没有覆盖全部绘图字符的中文字体：" + json.dumps(attempts, ensure_ascii=False))


def layout_audit(fig) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer = fig.bbox
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < outer.x0 - 1 or box.y0 < outer.y0 - 1 or
                                        box.x1 > outer.x1 + 1 or box.y1 > outer.y1 + 1):
            outside.append(text.get_text())
    legend_overlaps = []
    legend_texts = [text for legend in fig.legends for text in legend.findobj(Text)
                    if text.get_visible() and text.get_text()]
    title_overlaps = []
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()):
            if not label.get_visible() or not label.get_text():
                continue
            box = label.get_window_extent(renderer)
            for text in legend_texts:
                other = text.get_window_extent(renderer)
                if (min(box.x1, other.x1) - max(box.x0, other.x0) > .5
                        and min(box.y1, other.y1) - max(box.y0, other.y0) > .5):
                    legend_overlaps.append(dict(axis_label=label.get_text(), legend_label=text.get_text()))
        if fig._suptitle is not None and ax.title.get_visible() and ax.title.get_text():
            box, other = ax.title.get_window_extent(renderer), fig._suptitle.get_window_extent(renderer)
            if (min(box.x1, other.x1) - max(box.x0, other.x0) > .5
                    and min(box.y1, other.y1) - max(box.y0, other.y0) > .5):
                title_overlaps.append(dict(panel_title=ax.title.get_text(), figure_title=fig._suptitle.get_text()))
    return dict(text_bounds_status="PASS" if not outside and not legend_overlaps and not title_overlaps else "FAIL",
                text_outside_canvas=outside, axis_label_legend_text_overlaps=legend_overlaps,
                panel_title_figure_title_overlaps=title_overlaps,
                note="Canvas bounds, axes-text/figure-legend and panel/figure-title checks; visual review remains separate.")


def replot(analysis: Path, output: Path, font_override: Path | None = None) -> dict:
    analysis, output = analysis.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("输出目录已存在；不得改写已有图表：" + str(output))
    raw = analysis.read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    runs = validate_report(report)
    styles = identity_styles(runs)
    # Use a Chinese font before the first layout draw, then audit the exact texts.
    seed_required = {ord(char) for char in "中文身份目标返回测量起点属性拒绝0123456789−"}
    font_path, _ = choose_font(seed_required, font_override)
    font_manager.fontManager.addfont(str(font_path))
    prop = font_manager.FontProperties(fname=str(font_path))
    plt.rcParams.update({"font.family": prop.get_name(), "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True})
    timeline, timeline_data = timeline_figure(runs, styles)
    counts, count_data = count_figure(runs, styles)
    active, active_data = active_stall_figure(runs)
    named_figures = [("chain_identity_transport_zh", timeline), ("measured_identity_command_counts_zh", counts)]
    if active is not None:
        named_figures.append(("active_command_lease_clear_zh", active))
    figures = [fig for _, fig in named_figures]
    strings = figure_texts(figures)
    required = {ord(char) for value in strings for char in value if not char.isspace()}
    checked_font, font_receipt = choose_font(required, font_path)
    if checked_font != font_path:
        raise ValueError("字体覆盖检查改变字体，拒绝保存")
    layout = [layout_audit(fig) for fig in figures]
    if any(row["text_bounds_status"] != "PASS" for row in layout):
        raise ValueError("标签超出画布或图例重叠：" + json.dumps(layout, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for name, fig in named_figures:
        for suffix in ("png", "pdf"):
            path = output / (name + "." + suffix)
            fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=.12, facecolor="white")
            artifacts.append(path)
        plt.close(fig)
    metadata = dict(kind="T1_CHINESE_REPLOT_DATA_MAPPING", source_analysis_kind=report.get("kind"),
        new_experiment=False, formal_efficacy_results=False, identity_status_labels={key: value[1] for key, value in styles.items()},
        timeline=timeline_data, measured_counts=count_data, active_stall=active_data,
        input_runs_without_saved_frames=[run.get("session_id") for run in report["runs"] if not run.get("frame_rows")],
        clock_scope="Saved host monotonic receipts; missing slots are marked at nominal deadlines, not invented capture times.",
        return_scope="Backend return and source attribute acceptance do not establish adoption or correct instance identity.",
        source_attribute_marker_scope="The return endpoint is marked by the request's eventual attribute outcome; no distinct attribute audit clock is inferred.",
        review_commit_marker_scope="REVIEW_COMMITTED output-frame readbacks only, not the exact adoption time; repeated CHAIN_LOST event records are not counted as independent losses.")
    metadata_path = output / "plot_data_mapping.json"
    save_json(metadata_path, metadata)
    artifacts.append(metadata_path)
    source = Path(__file__).resolve()
    snapshot = output / source.name
    shutil.copyfile(source, snapshot)
    artifacts.append(snapshot)
    if analysis.read_bytes() != raw:
        raise RuntimeError("绘图期间输入报告改变，拒绝生成完成凭据")
    receipt = dict(kind="T1_CHINESE_REPLOT_RECEIPT", created_utc=datetime.now(timezone.utc).isoformat(),
        new_experiment=False, formal_efficacy_results=False,
        input_analysis=dict(path=str(analysis), sha256=sha256_bytes(raw)),
        plotter_source=dict(path=str(source), sha256=digest(source), snapshot_path=str(snapshot), snapshot_sha256=digest(snapshot)),
        font=font_receipt, plotted_run_count=len(runs), saved_frame_count=sum(len(run["frame_rows"]) for run in runs),
        layout_checks=layout, visual_review=dict(status="PENDING", note="Inspect the generated PNG files before delivery."),
        artifacts=[dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size) for path in artifacts],
        input_report_unchanged=True, overwrite_existing_output=False,
        limitations=["Chinese redraw of existing saved experiments; no new flight or cloud request.",
                     "Runtime completion, independent identity qualification and formal research efficacy are separate.",
                     "No inference of evidence adoption from backend return or source attribute acceptance."])
    save_json(output / "artifact_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="保存的独立分析 report.json")
    parser.add_argument("--output", type=Path, required=True, help="必须尚不存在的新输出目录")
    parser.add_argument("--font", type=Path, help="可选中文字体文件；默认查找 Windows 中文字体")
    args = parser.parse_args()
    result = replot(args.analysis, args.output, args.font)
    print(json.dumps(dict(output=str(args.output.resolve()), plotted_runs=result["plotted_run_count"],
                          saved_frames=result["saved_frame_count"], glyph_coverage=result["font"]["glyph_coverage"]), ensure_ascii=False))

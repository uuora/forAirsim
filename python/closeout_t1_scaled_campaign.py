"""Update existing thesis/status files from audited scales, incomplete attempts and Qt QA."""
import argparse
from datetime import datetime,timezone
import json,re
from pathlib import Path

from analyze_t1_identity_flight import digest

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def closeout(analysis,figures,layout,folder):
    analysis,figures,layout,folder=[Path(p).resolve() for p in (analysis,figures,layout,folder)]
    if (folder/"receipt.json").exists():raise FileExistsError("SCALED_CLOSEOUT_ALREADY_WRITTEN")
    data=read(analysis);runs=data["runs"]
    qualified=[r for r in runs if r["independent_scale_qualification"].startswith("PASS")]
    completed=[r for r in runs if r["endpoint"]["net_xy_displacement_m"] is not None]
    incomplete=[r for r in runs if r not in qualified]
    fig=read(figures/"artifact_receipt.json");qt=read(layout)
    if (fig["input_analysis"]["sha256"]!=digest(analysis) or fig["font"]["glyph_coverage"]!="PASS"
        or fig["visual_review"]["status"]!="PASS_RENDERED_PNG_REVIEW"
        or any(x["text_bounds_status"]!="PASS" for x in fig["layout_checks"])):
        raise ValueError("CHINESE_SCALE_FIGURES_REQUIRE_VERIFIED_INPUT_AND_RENDERED_REVIEW")
    if (qt["status"]!="PASS_OFFSCREEN_SAVED_EVIDENCE_LAYOUT" or qt["analysis"]["sha256"]!=digest(analysis)
        or qt["visual_review"]["status"]!="PASS_RENDERED_PNG_REVIEW"):
        raise ValueError("CURRENT_QT_RENDER_REVIEW_REQUIRED")
    for row in fig["artifacts"]+qt["renders"]+qt["source_files"]:
        if digest(Path(row["path"]))!=row["sha256"]:raise ValueError("REVIEWED_ARTIFACT_CHANGED")
    for r in runs:
        for row in r["input_receipts"]:
            if digest(Path(row["path"]))!=row["sha256"]:raise ValueError("RUN_INPUT_CHANGED")
    for r in qualified:
        if not (r["base_evidence"]["fresh_ground_closure_verified"] and r["base_evidence"]["engine_closed_after_safe_completion"]):
            raise ValueError("QUALIFIED_RUN_REQUIRES_FRESH_GROUND_CLOSURE")
    before=read(folder/"before.json")
    intake_path=folder/"pre_closeout_existing_chapter_updates.json"
    intake=read(intake_path) if intake_path.exists() else {"files":[]}
    updates={r["path"]:r for r in intake["files"]}
    for row in before["files"]:
        prior=updates.get(row["path"])
        expected=row["sha256"]
        if prior:
            if prior["original_before_sha256"]!=expected or digest(Path(prior["snapshot_path"]))!=prior["sha256"]:
                raise ValueError("EXISTING_CHAPTER_UPDATE_INTAKE_CHANGED")
            expected=prior["sha256"]
        if digest(Path(row["path"]))!=expected:raise ValueError("CANONICAL_DOCUMENT_CHANGED_BEFORE_CLOSEOUT")
    declaration=read(folder/"predeclared_plans.json")
    if digest(Path(declaration["protocol_path"]))!=declaration["protocol_sha256"]:raise ValueError("DECLARED_PROTOCOL_CHANGED")
    for row in declaration["plans"]:
        if digest(Path(row["plan_path"]))!=row["sha256"]:raise ValueError("ORIGINAL_FROZEN_PLAN_CHANGED")
    tests=(folder/"final_tests.stderr.log").read_text(encoding="utf-8")
    match=re.search(r"Ran (\d+) tests in ([\d.]+)s",tests)
    if not match or int(match[1])<427 or not tests.rstrip().endswith("OK"):raise ValueError("REGRESSION_GATE_NOT_PASSED")
    idle=read(folder/"final_idle_snapshot.json")
    if idle["engine_process_count"]!=0 or idle["rpc_connection_count"]!=0:raise ValueError("FRESH_FINAL_IDLE_REQUIRED")
    diagnosis_path=folder/"zero_hold_scope_diagnosis_20261007.json"
    diagnosis=read(diagnosis_path)
    for key in ("source_flight_report","source_independent_analysis"):
        row=diagnosis[key]
        if digest(Path(row["path"]))!=row["sha256"]:raise ValueError("DIAGNOSIS_SOURCE_CHANGED")
    if not (diagnosis["configured_settlement_unchanged"] and diagnosis["future_receipt_correction_implemented_not_flown"]
        and diagnosis["old_trial_still_fails_declared_scale_and_complete_grid"]):raise ValueError("DIAGNOSIS_SCOPE_INVALID")
    runtime=[]
    for r in runs:
        raw=read(Path(r["run"])/"flight/report.json")
        for row in raw["configuration"]["source_inventory"]+raw["configuration"]["legacy_source_inventory"]:
            if digest(Path(row["path"]))!=row["sha256"] or digest(Path(row["snapshot_path"]))!=row["sha256"]:
                raise ValueError("FROZEN_RUNTIME_SOURCE_CHANGED")
        runtime.extend(raw["configuration"]["source_inventory"])
    labels={"zero":"零输出门控","scale50":"50毫米尺度","scale100":"100毫米尺度"}
    table=["| 条件/会话 | 保存/计划主帧 | 正确/无输出/未评分或其他 | 净XY位移（毫米） | 独立判定 | 原完整完成晚到 |",
        "|---|---:|---:|---:|---|---:|"]
    for r in runs:
        c=r["base_evidence"]["measured_identity_counts"];net=r["endpoint"]["net_xy_displacement_m"]
        label=labels[r["scaled_motion_condition"]["name"]]
        if "_rep" in r["session_id"]:label+="重复"+r["session_id"].split("_rep")[1].split("_")[0]
        numbers=f"{c.get('CORRECT',0)}/{c.get('NO_OUTPUT',0)}/{sum(v for k,v in c.items() if k not in ('CORRECT','NO_OUTPUT'))}" if c else "无评分"
        table.append(f"| {label} | {r['base_evidence']['measured_frame_count']}/{r['plan_check']['expected_main_slots']} | {numbers} | {net*1000:.3f}".rstrip() if net is not None else
            f"| {label} | {r['base_evidence']['measured_frame_count']}/{r['plan_check']['expected_main_slots']} | {numbers} | 未核实")
        table[-1]+=(f" | {'通过' if r in qualified else '失败或缺终态'} | {r['timing_summary']['historical_full']['counts'].get('LATE',0) if r in completed else '未完整评价'} |")
    table="\n".join(table)
    identity={}
    for r in completed:
        for k,v in r["base_evidence"]["measured_identity_counts"].items():identity[k]=identity.get(k,0)+v
    nmain=sum(r["base_evidence"]["measured_frame_count"] for r in completed)
    planned=sum(r["plan_check"]["expected_main_slots"] for r in completed)
    late=sum(r["timing_summary"]["historical_full"]["counts"].get("LATE",0) for r in completed)
    processing_late=sum(r["timing_summary"]["frame_processing"]["counts"].get("LATE",0) for r in completed)
    ninit=sum(r["base_evidence"].get("acquisition_frame_count",0) for r in completed)
    complete50=[r for r in qualified if r["scaled_motion_condition"]["name"]=="scale50"]
    complete100=[r for r in qualified if r["scaled_motion_condition"]["name"]=="scale100"]
    scales_complete=len(complete50)>=2 and bool(complete100)
    notice="50毫米两次完整试次和100毫米试次均通过受控尺度核查。" if scales_complete else "50毫米独立重复未达标，100毫米条件未执行。"
    evidence=f"[完整审计与中断记录]({analysis.as_posix()})"
    overview=(f"本轮共保留{len(runs)}个UE启动尝试，{len(qualified)}个完整受控会话通过核查；"
        f"有终态的{len(completed)}个会话保存主帧{nmain}/{planned}个，初始化另计{ninit}个，正确{identity.get('CORRECT',0)}帧、无输出{identity.get('NO_OUTPUT',0)}帧。"
        f"全部{len(incomplete)}个未通过核查的尝试单列，包括已安全收尾的未达标试次。已保存有终态主帧的原完整完成晚到{late}/{nmain}，帧处理晚到{processing_late}/{nmain}。 {evidence}")
    incomplete_text=("启动重复1的自有UE进程13732在启动18秒后发生访问违例；运行器没有主帧，也未获得安全收尾证明。"
        "重复2保存212帧主测量流水，自身状态证明已飞行，但原报告停留在RUNNING且无终态；这212帧仅用于查看，身份未独立评分，端点与安全收尾未核实。"
        "恢复工作时新查进程与RPC均已退出。两次原记录及重新声明的重复计划分别保留，不能以成功重试改写失败。")
    interpretation=("本轮160/320秒条件与50/100毫米判据均在飞行前声明。指令速度上限、租约、源图时效、实际清零预留和严格采样停稳保持原值。"
        "不同阶段时长形成不同执行尺度条件；位移增大不能解释为某方法更优。帧具有时间关联，试次是独立单位。"
        "固定高度XY仍未建立完整几何轨迹优化，原0.4秒完整完成晚到保持原判断。")
    diagnosis_text=("重复3正常安全收尾，但净XY位移46.583毫米低于预声明50毫米判据，400个计划主槽仅保存393帧，缺失7槽。"
        "独立零保持核查还在f240发现：最后五个历史样本中有一个位于已声明稳定窗口之前，仍带非零反馈指令。"
        "已实现只绑定原稳定窗口实际样本的证明修正，并用该实际记录及拒识样例完成四项回归；停稳阈值、稳定时长和收尾条件保持原值。"
        "此修正尚未接入冻结V9并重新飞行，旧试次的失败判定保留。"
        "本次51个已完成VLX请求的实际耗时中位数0.750秒、最大1.047秒，没有超过2秒；"
        "90个无输出主帧涉及等待语义返回、无有效复核及关联链缺口。记录要求进一步测量采集和本地处理成本，尚不能据此确认单一原因。"
        f"[逐帧诊断与原证据]({diagnosis_path.as_posix()})")
    next_steps=("下一步先修正零保持证明的采样范围并复验，定位0.4秒采样缺失与由此引起的证据链失效，重新冻结并完成50毫米重复与100毫米条件；"
        "再实现与同一身份/来源/控制约束相容的完整姿态及三维几何执行，记录观测质量、路径与失败；"
        "随后统一四方法执行接口，冻结场景、参数、种子与全启动回合指标，完成固定跟随、距离规划、可见性规划、身份复核联合规划及YOLO/VLX消融。"
        "任务一A3与A7—A10仍待补；任务二、三保持原六章安排。")
    doc=ROOT/"docs/research-plan-20261004/任务一尺度验证与Qt5工作台结果.md"
    if doc.exists():raise FileExistsError("RESULT_DOC_EXISTS")
    body=("# 任务一尺度验证与Qt5工作台结果\n\n整理更新：2026-10-07。记录始于2026-10-06，继续阶段采用2026-10-07的新会话编号。\n\n"
        "## 1. 实际结果\n\n"+overview+"\n\n"+table+"\n\n"+notice+"\n\n## 2. 失败、中断及边界\n\n"+incomplete_text+"\n\n"+interpretation+
        "\n\n"+diagnosis_text+"\n\n## 3. 中文图表\n\n")
    for row in fig["visual_review"]["reviewed_pngs"]:
        body+=f"![受控尺度验证中文图表]({Path(row['path']).as_posix()})\n\n"
    body+=("## 4. Qt5工作台\n\n已实现会话切换、逐槽图像回放、中文轨迹/身份/完成时序、播放暂停、PNG和CSV导出。"
        "气球专用YOLO与任务一使用同一冻结权重；一次真实单图VLX定位约0.908秒，外部候选框未上传。"
        "失败会话保留计划槽位；中断流水标为未评分，缺失终态不显示已核实位移。当前界面检查使用离屏Qt，原生桌面鼠标验收尚待补。"
        f"[实际单图与回放检查]({(ROOT/'logs/t1_qt_workbench_20261006/qt_verification.json').as_posix()}) · [当前布局复核]({layout.as_posix()}) · "
        "[使用说明](D:/forAirsim/docs/任务一Qt5实验工作台.md)\n\n"
        f"软件回归{match[1]}项通过（{match[2]}秒）；运行源码、历史十二模块与已审阅图件核对一致。新进程/RPC最终空闲检查另存。\n\n"
        "## 5. 下一步\n\n"+next_steps+"\n\nAI协助说明：AI辅助实现、检查和证据整理；结果来自实际记录，缺失证据保留未知，研究解释由作者审定。\n")
    doc.write_text(body,encoding="utf-8")
    fragment="\n### 2026-10-07 50/100毫米尺度与Qt5实验工作台\n\n"+overview+"\n\n"+notice+" "+interpretation+"\n\n"+incomplete_text+"\n\n"+diagnosis_text+"\n\n"+next_steps+f"\n\n[本轮结果与图表]({doc.as_posix()})\n"
    for name in ("docs/research-plan-20261004/硕士论文章节任务与图表安排.md","docs/research-plan-20261004/任务一论文内容与完成标准.md"):
        path=ROOT/name;path.write_text(path.read_text(encoding="utf-8")+fragment,encoding="utf-8")
    path=ROOT/"docs/thesis/第三章_单机感知轨迹优化.md";text=path.read_text(encoding="utf-8")
    addition=fragment.replace(notice+" "+interpretation, "表3-10 预声明尺度、完整与未完成试次。\n\n"+table+"\n\n"+notice+" "+interpretation)
    marker="### AI协助说明";assert text.count(marker)==1
    path.write_text(text.replace(marker,addition+"\n"+marker),encoding="utf-8")
    path=ROOT/"docs/thesis/第二章_问题建模与仿真平台.md";text=path.read_text(encoding="utf-8");assert text.count(marker)==1
    platform=("### 2026-10-07 预声明尺度与证据回放\n\n"+interpretation+" "+evidence+"\n\n"
        "Qt5回放使用分析、原报告和RGB校验值绑定来源，初始化与主帧分别记账。失败未采集槽位保留，缺少实际时钟时显示计划时刻；中断流水明确未评分。"
        "缺少终态的试次没有已核实端点，单图云端请求仍为图像＋文字。"
        "[界面与核查说明](D:/forAirsim/docs/任务一Qt5实验工作台.md)\n\n")
    path.write_text(text.replace(marker,platform+marker),encoding="utf-8")
    path=ROOT/"README.md";text=path.read_text(encoding="utf-8")
    headline="**最新推进（2026-10-07，尺度验证与Qt5工作台）**："+notice+f"保留全部{len(runs)}个尝试，完整会话保存{nmain}主帧，失败/中断单列；{match[1]}项回归通过。"
    headline+=f"Qt5现支持中文实验回放、图表与导出，并保留真实YOLO/VLX单图验证。完整几何和正式对照仍待完成。[结果与下一步]({doc.as_posix()}) · [启动界面](D:/forAirsim/Run-VLX-GUI.cmd)。\n\n"
    first,rest=text.split("\n",1);path.write_text(first+"\n\n"+headline+rest.lstrip("\n"),encoding="utf-8")
    metadata=dict(date="2026-10-07",status="SCALED_DEVELOPMENT_MEASURED_FORMAL_T1_PENDING",analysis_path=str(analysis),analysis_sha256=digest(analysis),
        report_path=str(doc),figure_receipt_path=str(figures/"artifact_receipt.json"),all_attempts=len(runs),complete_qualified_sessions=len(qualified),
        incomplete_sessions=[r["session_id"] for r in incomplete],complete_main_frames=nmain,complete_planned_slots=planned,complete_acquisition_frames=ninit,
        complete_main_identity_counts=identity,complete_late_frames=late,frame_processing_late_frames=processing_late,completed_terminal_sessions=len(completed),
        scale_50_repeated_and_100_passed=scales_complete,regression_tests=int(match[1]),formal_efficacy_results=False,whole_thesis_complete=False)
    gui=dict(status="SAVED_REPLAY_AND_SINGLE_IMAGE_VERIFIED_OFFSCREEN",entrypoint="Run-VLX-GUI.cmd",guide="docs/任务一Qt5实验工作台.md",
        real_single_image_receipt="logs/t1_qt_workbench_20261006/qt_verification.json",current_layout_receipt=str(layout),current_layout_sha256=digest(layout),
        native_desktop_interaction_tested=False,live_flight_control_implemented=False,cloud_request_mode="image_text",proposals_transmitted=False)
    protocol_path=ROOT/"configs/t1_completion_protocol_v1.json";protocol=read(protocol_path)
    protocol["latest_scaled_motion_development"]=metadata;protocol["qt_workbench"]=gui;save(protocol_path,protocol)
    tasks_path=ROOT/"configs/thesis_simulation_tasks.json";tasks=read(tasks_path);old_other=[t for t in tasks["tasks"] if t["id"]!="T1"]
    task=next(t for t in tasks["tasks"] if t["id"]=="T1");task.update(status=metadata["status"],latest_scaled_motion_development=metadata,qt_workbench=gui,
        next_gate="full_pose_geometry_execution_then_formal_methods_and_ablations" if scales_complete else "fix_zero_hold_proof_scope_and_sampling_then_50mm_repeat_and_100mm")
    assert old_other==[t for t in tasks["tasks"] if t["id"]!="T1"];tasks["updated"]="2026-10-07";save(tasks_path,tasks)
    save(folder/"receipt.json",dict(kind="T1_SCALED_CAMPAIGN_QT_CLOSEOUT",created_utc=datetime.now(timezone.utc).isoformat(),metadata=metadata,gui=gui,
        analysis_sha256=digest(analysis),tests=int(match[1]),before_files=before["files"],updated_files=[dict(path=row["path"],sha256=digest(Path(row["path"]))) for row in before["files"]],
        diagnosis=dict(path=str(diagnosis_path),sha256=digest(diagnosis_path),future_receipt_correction_not_flown=True),
        existing_chapter_updates_intake=dict(path=str(intake_path),sha256=digest(intake_path)) if intake_path.exists() else None,
        result_doc=dict(path=str(doc),sha256=digest(doc)),runtime_source_count=len({r['path'] for r in runtime}),final_idle_path=str(folder/"final_idle_snapshot.json"),formal_efficacy_results=False))
    print("closeout",len(runs),"attempts",len(qualified),"qualified",match[1],"tests",notice)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("analysis","figures","layout","folder"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();closeout(a.analysis,a.figures,a.layout,a.folder)

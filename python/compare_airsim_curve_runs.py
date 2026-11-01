"""Compare selected frozen AirSim execution records without simulator calls.

Strictly reject different physical/control/source-plan conditions. Show every
requested row, workflow/control outcomes and missing execution slots. Numeric
summaries are explicitly conditional on valid completed records, including
control-gate failures; no significance or confidence intervals are estimated.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from audit_airsim_curve_replay_pilot import jsonlines, verify_loaded_profile, verify_trace


ROOT=Path(__file__).resolve().parents[1]
DISPLAY_KEYS={"ViewMode","EngineSound","LogMessagesVisible","RecordUIVisible","Recording"}
EXECUTION_METRICS=("position_rmse_m","terminal_position_error_m","reference_duration_s",
         "max_position_error_m","terminal_velocity_error_mps","quality_term",
         "speed_clipped_time_fraction","command_slew_clipped_time_fraction")
PLANNING_METRICS=('planning_search_wall_s','planning_validation_wall_s','planning_search_plus_validation_wall_s',
                  'network_generation_s','network_update_s','full_surrogate_calls')
METRICS=EXECUTION_METRICS+PLANNING_METRICS
CONTROL_FUNCTIONS=("limit_command","read_state","initialize_curve","metrics_from_trace",
                   "new_execution_collision","replay_one")
SCOPE=("Read-only comparison of selected recorded AirSim replays, not new flights or AirSim optimization. "
       "Workflow completion and control-error success are separate. Descriptive statistics only; "
       "no significance/CI, causal claim, curve-basis superiority or wall-clock speed claim. "
       "Pure display settings are recorded but excluded from physical/control comparability.")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def replay_directory(path):
    path=Path(path).resolve()
    if (path/"replay"/"manifest.json").is_file():
        path=path/"replay"
    if not (path/"manifest.json").is_file():
        raise ValueError(f"缺少回放manifest：{path}")
    return path


def safe_child(root,relative):
    target=(root/relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"证据路径越界：{relative}")
    return target


def snapshot_functions(path):
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    functions={n.name:stable_hash(ast.dump(n,include_attributes=False)) for n in tree.body
               if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in (*CONTROL_FUNCTIONS,'tracking_command','plan_curve','setup_targets')}
    if not set(CONTROL_FUNCTIONS)<=set(functions):
        raise ValueError("回放源码快照缺少必要控制/记录函数，不能确认可比性")
    return functions


def inspect_run(path):
    directory=replay_directory(path)
    manifest_path=directory/"manifest.json"
    manifest=load(manifest_path)
    if manifest.get("probe_only") or manifest.get("status")=="RUNNING":
        raise ValueError(f"批次尚未结束或只有地面探针，不能当作执行比较：{directory}")
    config=manifest["config"]
    all_plans=manifest["plans_document"]["plans"]
    plan_map={p["group_id"]:p for p in all_plans}
    if len(plan_map)!=len(all_plans):
        raise ValueError("计划包含重复group_id")
    scheduled=manifest.get("selected_group_ids",[p["group_id"] for p in all_plans])
    if len(set(scheduled))!=len(scheduled) or any(g not in plan_map for g in scheduled):
        raise ValueError("manifest选择组无效")
    if manifest["planned_replays"]!=len(scheduled):
        raise ValueError("计划槽位数量和选择组不一致")
    input_hashes={str(manifest_path):sha(manifest_path)}
    plan_sha=None
    core_sha={}
    replay_functions=None
    config_bound=False
    for source in manifest["sources"]:
        snapshot=safe_child(directory,source["snapshot"])
        if sha(snapshot)!=source["sha256"]:
            raise ValueError(f"源码/配置快照hash错误：{snapshot}")
        input_hashes[str(snapshot)]=source["sha256"]
        name=Path(source["path"]).name
        if name=="airsim_replay_plan.json" or (name.endswith('.json') and load(snapshot)==manifest['plans_document']):
            if load(snapshot)!=manifest["plans_document"]:
                raise ValueError("内嵌计划与原计划快照不一致")
            plan_sha=source["sha256"]
        elif name in ("parameterizations.py","curves.py"):
            core_sha[name]=source["sha256"]
            current=ROOT/"python/trajectory_research"/name
            if not current.is_file() or sha(current)!=source["sha256"]:
                raise ValueError("当前曲线核查实现与冻结源码不同，请先核对该版本")
        elif name=="run_airsim_curve_replay_pilot.py":
            replay_functions=snapshot_functions(snapshot)
        elif name=='curve_airsim_sphere_scene.py':
            core_sha[name]=source['sha256']
        elif name.endswith(".json") and load(snapshot)==config:
            config_bound=True
    if plan_sha is None or not config_bound or not {"parameterizations.py","curves.py"}<=set(core_sha) or replay_functions is None:
        raise ValueError("缺少绑定的计划/配置/曲线或回放源码快照，无法建立可比条件")
    launcher_path=directory.parent/"launcher_manifest.json"
    launcher=load(launcher_path) if launcher_path.is_file() else {}
    if launcher:
        input_hashes[str(launcher_path)]=sha(launcher_path)
    map_name=launcher.get("map",config.get("map"))
    if not map_name or (launcher.get("map") and config.get("map") and launcher["map"]!=config["map"]):
        raise ValueError("地图/物理场景元数据缺失或不一致")
    probe_path=directory/"ground_probe.json"
    if not probe_path.is_file():
        raise ValueError("缺少实际加载settings的地面探针记录，无法确认backend")
    probe=load(probe_path)
    verify_loaded_profile(config,probe)
    input_hashes[str(probe_path)]=sha(probe_path)
    loaded=probe["loaded_settings"]
    settings={key:value for key,value in loaded.items() if key not in DISPLAY_KEYS}
    runtime_hashes={Path(r["path"]).name:r["sha256"] for r in launcher.get("runtime_files",[])}
    conditions=dict(map=map_name,physical_settings=settings,controller=config["controller"],
        initialization=config["initialization"],execution=config["execution"],
        control_error_gates=config["control_error_gates"],source_plan_sha256=plan_sha,
        curve_source_sha256=core_sha,replay_control_function_sha256=replay_functions,
        runtime_binary_sha256=runtime_hashes,rpc_timeout_s=config.get("rpc_timeout_s"),
        lifecycle_rpc_timeout_s=config.get("lifecycle_rpc_timeout_s"))
    conditions['sphere_scene']=config.get('sphere_scene')
    conditions['protocol_revision']=config.get('protocol_revision',1)
    conditions['target_scene']=config.get('target_scene')
    rows_path=directory/"rows.json"
    reports=load(rows_path) if rows_path.is_file() else []
    if rows_path.is_file():
        input_hashes[str(rows_path)]=sha(rows_path)
    attempted=[r["group_id"] for r in reports]
    if attempted!=scheduled[:len(attempted)] or len(attempted)!=len(set(attempted)):
        raise ValueError("执行结果不对应预定选择槽位的前缀")
    completed=sum(r["status"]=="COMPLETED" for r in reports)
    if manifest.get("attempted_replays")!=len(reports) or manifest.get("completed_replays")!=completed:
        raise ValueError("manifest完成计数与实际report不一致")
    if manifest.get("not_run_group_ids")!=scheduled[len(reports):]:
        raise ValueError("manifest未运行槽位与实际结果不一致")
    label=directory.parent.name if directory.name=="replay" else directory.name
    return dict(directory=directory,manifest=manifest,config=config,plans=plan_map,scheduled=scheduled,
        reports={r["group_id"]:r for r in reports},input_hashes=input_hashes,conditions=conditions,
        fingerprint=stable_hash(conditions),run_label=label,run_id=f"{label}-{stable_hash(str(directory))[:8]}",
        display_mode=loaded.get("ViewMode"),backend=f"AirSim/{loaded['PhysicsEngineName']}/{loaded['Vehicles'][config['vehicle']]['VehicleType']}")


def collect_record(run,group_id):
    if group_id not in run["plans"]:
        raise ValueError(f"批次计划中不存在所选组：{group_id}")
    plan=run["plans"][group_id]
    report=run["reports"].get(group_id)
    planned=group_id in run["scheduled"]
    row=dict(run_id=run["run_id"],run_label=run["run_label"],run_path=str(run["directory"]),group_id=group_id,
        seed=plan["seed"],representation=plan["representation"],family=plan["family"],plan_parameters=plan["parameters"],
        planned_duration_s=plan["parameters"][-1],planned_slot=planned,actual_backend=run["backend"],
        map=run["conditions"]["map"],display_mode=run["display_mode"],condition_fingerprint=run["fingerprint"],
        source_plan_sha256=run["conditions"]["source_plan_sha256"],selection_provenance=plan.get("selection_provenance",{}),
        control_error_gates=run["config"]["control_error_gates"],status="NOT_RUN" if planned else "NOT_SELECTED",
        producer_status=None,workflow_completed=False,data_quality_ok=None,control_error_gates_passed=None,
        failure_reason=None,data_quality_error=None,duplicate_evidence_of=None,included_in_numeric_summary=False,
        report_sha256=None,trace_sha256=None,**{key:None for key in METRICS})
    if report is None:
        return row
    if not planned or report["plan"]!=plan:
        raise ValueError("实际执行结果与所选源计划不一致")
    folder=safe_child(run["directory"],group_id)
    report_path,trace_path,event_path=(folder/name for name in ("report.json","trace.json","events.jsonl"))
    if load(report_path)!=report:
        raise ValueError("rows与组内report副本不一致")
    trace=load(trace_path)
    for path in (report_path,trace_path,event_path):
        run["input_hashes"][str(path)]=sha(path)
    row.update(producer_status=report["status"],workflow_completed=report["status"]=="COMPLETED",
        failure_reason=report.get("failure_reason"),report_sha256=sha(report_path),trace_sha256=sha(trace_path))
    planning=report.get('planning_evidence') or {}
    if planning:
        if planning!=run['manifest'].get('planning_records',{}).get(group_id):raise ValueError('规划计时副本不一致')
        bound=[s for s in run['manifest']['sources'] if s['sha256']==planning['source_sha256']]
        if len(bound)!=1:raise ValueError('缺少原规划结果快照')
        model=load(safe_child(run['directory'],bound[0]['snapshot']))
        if model['counts']!=planning['counts'] or model['timings']!=planning['timings']:raise ValueError('规划调用数或计时与冻结结果不符')
    timings=planning.get('timings',{});counts=planning.get('counts',{})
    row.update(planning_search_wall_s=timings.get('search_wall_s'),planning_validation_wall_s=timings.get('selected_validation_wall_s'),
        planning_search_plus_validation_wall_s=timings.get('search_plus_selected_validation_wall_s'),
        network_generation_s=timings.get('neural_generation_s'),network_update_s=timings.get('neural_update_s'),
        full_surrogate_calls=counts.get('total_objective_calls'),
        planning_model_revision=planning.get('model_revision'),
        planning_scope=planning.get('scope','unavailable in this record'),
        airsim_replay_attempts=1,complete_airsim_reference_executions=int(report['status']=='COMPLETED'),
        execution_host_wall_s=report.get('reference_execution_wall_s'),task_success=report.get('metrics',{}).get('task_success'))
    try:
        audited=verify_trace(plan,trace,report,run["config"],jsonlines(event_path))
    except (AssertionError,ValueError,KeyError,IndexError,TypeError) as exc:
        row.update(status="DATA_QUALITY_FAILED",data_quality_ok=False,
                   data_quality_error=f"{type(exc).__name__}: {exc}")
        return row
    row.update(status=audited["classification"],data_quality_ok=True,
               _trace=trace,_plan=plan,_audit=audited)
    if row["workflow_completed"] and audited["metrics_recomputed"]:
        metrics=audited["recomputed_metrics"]
        row.update({name:metrics[name] for name in EXECUTION_METRICS})
        row["control_error_gates_passed"]=metrics["control_error_gates_passed"]
        row["included_in_numeric_summary"]=True
    return row


def summarize(records):
    groups=[]
    for group_id in dict.fromkeys(r["group_id"] for r in records):
        rows=[r for r in records if r["group_id"]==group_id]
        numeric=[r for r in rows if r["included_in_numeric_summary"]]
        statistics={}
        for key in METRICS:
            values=np.array([r[key] for r in numeric if r.get(key) is not None and np.isfinite(r[key])],dtype=float)
            statistics[key]=dict(n=len(values),mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values)>1 else None,
                min=float(values.min()) if len(values) else None,max=float(values.max()) if len(values) else None)
        groups.append(dict(group_id=group_id,n_rows=len(rows),n_scheduled=sum(r["planned_slot"] for r in rows),
            n_attempted=sum(r["producer_status"] is not None for r in rows),
            n_workflow_completed=sum(r["workflow_completed"] for r in rows),n_valid_completed=len(numeric),
            n_control_pass=sum(r["control_error_gates_passed"] is True for r in numeric),
            n_control_fail=sum(r["control_error_gates_passed"] is False for r in numeric),
            n_execution_failed=sum(r["producer_status"]=="FAILED" for r in rows),
            n_not_run=sum(r["status"]=="NOT_RUN" for r in rows),
            n_not_selected=sum(r["status"]=="NOT_SELECTED" for r in rows),
            n_data_quality_failed=sum(r["status"]=="DATA_QUALITY_FAILED" for r in rows),
            n_duplicate_evidence=sum(r["duplicate_evidence_of"] is not None for r in rows),
            statistics=statistics))
    return groups


def plot_comparison(output,records,groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
    folder=output/"figures"
    folder.mkdir()
    fig=plt.figure(figsize=(12,8),layout="constrained")
    ax=fig.add_subplot(111,projection="3d")
    fig2,ax2=plt.subplots(figsize=(12,6),layout="constrained")
    plotted=0
    multiple_runs=len({r["run_id"] for r in records})>1
    for i,row in enumerate(records):
        trace=row.get("_trace")
        if not trace or not trace["states"]:
            continue
        states=trace["states"]
        q,p=(np.array([s[key] for s in states]) for key in ("reference_position_zup_m","position_zup_m"))
        t=np.array([s["reference_elapsed_s"] for s in states])
        label=row["group_id"]+(f" | {row['run_label']}" if multiple_runs else "")
        if not row["workflow_completed"]:
            label+=" [FAILED]"
        color=plt.cm.tab20(i%20)
        ax.plot(*q.T,color=color,ls="--",alpha=.4)
        ax.plot(*p.T,color=color,label=label)
        keep=t<=row["planned_duration_s"]
        ax2.plot(t[keep],np.linalg.norm(q[keep]-p[keep],axis=1),color=color,label=label)
        plotted+=1
    ax.set(xlabel="Local x (m)",ylabel="Local y (m)",zlabel="Height z-up (m)",
        title="Selected AirSim records: dashed reference, solid actual")
    ax2.set(xlabel="Actual simulation reference time (s)",ylabel="Synchronous sampled position error (m)",
        title="Selected AirSim records (control-gate failures retained)")
    if records:
        ax2.axhline(records[0]["control_error_gates"]["sampled_peak_position_m"],color="black",ls=":",label="Peak-error gate")
    if plotted:
        ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1.))
        ax2.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1.))
    else:
        ax.text2D(.1,.5,"No usable execution trace",transform=ax.transAxes)
        ax2.text(.1,.5,"No usable execution trace",transform=ax2.transAxes)
    fig3,axes=plt.subplots(1,3,figsize=(15,5.5),layout="constrained")
    names=[g["group_id"] for g in groups]
    for axis,key,title in zip(axes,("position_rmse_m","terminal_position_error_m","reference_duration_s"),
                              ("Position RMSE (m)","Terminal position error (m)","Reference duration T (s)")):
        for i,group in enumerate(groups):
            stat=group["statistics"][key]
            if stat["mean"] is not None:
                axis.bar(i,stat["mean"],color=plt.cm.tab20(i%20),alpha=.55)
                values=[r[key] for r in records if r["group_id"]==group["group_id"] and r["included_in_numeric_summary"]]
                jitter=np.linspace(-.10,.10,len(values)) if len(values)>1 else [0.]
                axis.scatter(i+np.array(jitter),values,color="black",s=16,zorder=3)
                if stat["std"] is not None:
                    axis.errorbar(i,stat["mean"],yerr=stat["std"],fmt="none",color="black",capsize=4)
                axis.annotate(f"n={stat['n']}; pass={group['n_control_pass']}",(i,stat["mean"]),
                              xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
            else:
                axis.text(i,.02,"NO VALID\nCOMPLETION",ha="center",transform=axis.get_xaxis_transform(),fontsize=7)
        axis.set(title=title,xticks=range(len(names)),xticklabels=names)
        axis.tick_params(axis="x",labelrotation=35,labelsize=8)
        axis.grid(axis="y",alpha=.2)
    fig3.suptitle("Valid completed records only, including control failures; bars=mean, whiskers=sample SD (n>1)")
    for name,figure in (("trajectories",fig),("tracking_errors",fig2),("metrics",fig3)):
        figure.savefig(folder/f"{name}.png",dpi=170)
        figure.savefig(folder/f"{name}.pdf")
        plt.close(figure)


def write_report(output,summary):
    def number(value):
        return "—" if value is None else f"{value:.5g}"
    lines=["# 用户所选 AirSim 回放结果比较","",f"条件指纹：`{summary['condition_fingerprint']}`。",
        "","比较只读取已有记录，没有重跑仿真。流程完成与控制判据通过分别统计；不同物理/控制条件默认拒绝合并。",
        "纯展示设置（窗口/视角、声音、录制UI等）不进入物理条件指纹，但逐条保留；本报告不据此比较墙钟速度。",
        "","| 方法组 | 所选行/预定槽位 | 流程完成 | 有效完成 | 控制通过/失败 | 执行失败 | 未运行 | 数据质量失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        lines.append(f"| {group['group_id']} | {group['n_rows']}/{group['n_scheduled']} | {group['n_workflow_completed']} | "
            f"{group['n_valid_completed']} | {group['n_control_pass']}/{group['n_control_fail']} | {group['n_execution_failed']} | "
            f"{group['n_not_run']} | {group['n_data_quality_failed']} |")
    lines += ["","以下均值是**有效完成记录中的条件性描述**，包含控制门槛未通过的记录。失败、未运行和数据质量失败没有填造分数，已在上表及逐条CSV保留。不能只根据这些均值概括总体成功率。",
        "样本标准差仅在 n>1 时计算；n=1 时不报告显著性、置信区间或方法优越性。同一原始证据的复制另行标识，不当作独立重复。",
        "","| 方法组 | 数值样本n | RMSE 均值/标准差 (m) | 终端位置误差 均值/标准差 (m) | 参考T 均值/标准差 (s) |",
        "|---|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        cells=[f"{number(group['statistics'][key]['mean'])} / {number(group['statistics'][key]['std'])}" for key in
               ("position_rmse_m","terminal_position_error_m","reference_duration_s")]
        lines.append(f"| {group['group_id']} | {group['n_valid_completed']} | "+" | ".join(cells)+" |")
    lines += ["","参考执行时长T来自已选曲线，不能解释成规划计算耗时；API请求和返回时间不是电机执行确认。",
        "Bézier/Hermite 可以表示同一条物理曲线，回放差异还包含初始化、仿真时序等因素；本比较不把换基当成独立方法创新。",
        "","## 逐条选择与状态","","| 批次 | 方法组 | 状态 | 显示模式 | 原因/重复证据 |","|---|---|---|---|---|"]
    lines[-4:-4]=['','规划数据来自冻结的简化模型搜索证据，并非本次AirSim在线搜索；相同方案多次回放不构成新的独立规划计时。',
        '首次网络导入、初始化等实际开销保留；当前为等候选数开发实验，不能据此声称墙钟提效。',
        '', '| 方法 | 搜索秒 | 完整验证秒 | 网络生成秒 | 网络更新秒 | 完整简化模型调用 |', '|---|---:|---:|---:|---:|---:|']+[
        '| '+g['group_id']+' | '+' | '.join(number(g['statistics'][k]['mean']) for k in ('planning_search_wall_s','planning_validation_wall_s','network_generation_s','network_update_s','full_surrogate_calls'))+' |' for g in summary['groups']]+['']
    for row in summary["records"]:
        note=row.get("data_quality_error") or row.get("failure_reason") or row.get("duplicate_evidence_of") or ""
        note=str(note).replace("|","/").replace("\n"," ")
        lines.append(f"| {row['run_id']} | {row['group_id']} | {row['status']} | {row['display_mode']} | {note} |")
    lines += ["","输出：comparison.csv、summary.json、source_hashes.json，以及 figures/ 中轨迹、误差和指标的 PNG/PDF。",
        "AI 辅助完成比较器与说明。原始证据保持不变；统计为描述性汇总，未进行独立性或统计功效论证。",""]
    (output/"comparison.md").write_text("\n".join(lines),encoding="utf-8")


def build_comparison(run_paths,output,group_ids=None,*,selections=None):
    """Create a new comparison folder; never alter source evidence.

    Legacy form: build_comparison([run,...], out, [group,...]).
    Exact rows: build_comparison([], out, selections=[{'run_path':..., 'group_ids':[...]}]).
    With selections, only those pairs are included; run_paths is not expanded.
    """
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"比较输出目录已存在：{output}")
    requests=[]
    if selections is not None:
        if group_ids is not None:
            raise ValueError("精确selections模式不能同时传全局group_ids")
        if not isinstance(selections,list) or not selections:
            raise ValueError("selections必须为非空列表")
        for selection in selections:
            ids=selection.get("group_ids")
            if not isinstance(ids,list) or not ids:
                raise ValueError("每条selection需要非空group_ids")
            requests.append((selection["run_path"],ids))
    else:
        if not run_paths:
            raise ValueError("请选择至少一个已结束批次")
        requests=[(path,group_ids) for path in run_paths]
    run_cache={}
    pairs=[]
    seen=set()
    for path,ids in requests:
        directory=replay_directory(path)
        key=str(directory)
        if key not in run_cache:
            run_cache[key]=inspect_run(directory)
        run=run_cache[key]
        selected=run["scheduled"] if ids is None else ids
        if not selected or len(set(selected))!=len(selected):
            raise ValueError("组选择不能为空或重复")
        for group_id in selected:
            pair=(key,group_id)
            if pair in seen:
                raise ValueError("同一批次/组被重复选择，不能伪作独立重复")
            seen.add(pair)
            pairs.append((run,group_id))
    conditions=list(run_cache.values())
    first=conditions[0]
    for run in conditions[1:]:
        if run["fingerprint"]!=first["fingerprint"]:
            changed=[key for key in first["conditions"] if first["conditions"][key]!=run["conditions"].get(key)]
            raise ValueError("批次条件不可合并；请分别比较。差异字段："+", ".join(changed))
    records=[collect_record(run,group_id) for run,group_id in pairs]
    evidence_seen={}
    for row in records:
        if row["report_sha256"] is None:
            continue
        execution_key=(row["group_id"],row["report_sha256"],row["trace_sha256"])
        if execution_key in evidence_seen:
            row["duplicate_evidence_of"]=evidence_seen[execution_key]
            row["included_in_numeric_summary"]=False
        else:
            evidence_seen[execution_key]=f"{row['run_id']}/{row['group_id']}"
    groups=summarize(records)
    public_records=[{key:value for key,value in row.items() if not key.startswith("_")} for row in records]
    summary=dict(schema_version=1,scope=SCOPE,condition_fingerprint=first["fingerprint"],
        comparable_conditions=first["conditions"],selected_records=len(records),input_runs=len(run_cache),
        groups=groups,records=public_records,statistic_scope="Conditional valid-completion summaries; includes control "
            "failures, excludes missing/failed/invalid scores with explicit outcome counts. Sample SD only n>1; no CI/significance.")
    output.mkdir(parents=True,exist_ok=False)
    save(output/"summary.json",summary)
    fields=list(public_records[0])
    with (output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for row in public_records:
            writer.writerow({key:json.dumps(value,ensure_ascii=False,allow_nan=False) if isinstance(value,(list,dict)) else value
                             for key,value in row.items()})
    write_report(output,summary)
    plot_comparison(output,records,groups)
    input_hashes={path:value for run in run_cache.values() for path,value in run["input_hashes"].items()}
    source_paths=[Path(__file__),ROOT/"python/audit_airsim_curve_replay_pilot.py",
                  ROOT/"python/trajectory_research/parameterizations.py",ROOT/"python/trajectory_research/curves.py"]
    hashes=dict(input_files_sha256=input_hashes,comparator_sources_sha256={str(path):sha(path) for path in source_paths},
        output_files_sha256={path.relative_to(output).as_posix():sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        scope="Hashes bind consumed evidence and generated artifacts; no edits to source evidence.")
    save(output/"source_hashes.json",hashes)
    return dict(status="COMPLETE",output_dir=str(output),comparison_csv=str(output/"comparison.csv"),
        summary_json=str(output/"summary.json"),comparison_md=str(output/"comparison.md"),
        selected_records=len(records),groups=len(groups),condition_fingerprint=first["fingerprint"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs",nargs="+",type=Path)
    mode.add_argument("--selection-file",type=Path,help="JSON list, or {'selections':[{'run_path':..., 'group_ids':[...]},...]}")
    parser.add_argument("--groups",nargs="+")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    selections=None
    if args.selection_file:
        if args.groups:
            parser.error("--groups只适用于--runs；精确选择写在selection-file中")
        document=load(args.selection_file)
        selections=document.get("selections") if isinstance(document,dict) else document
    result=build_comparison(args.runs or [],args.output,args.groups,selections=selections)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()

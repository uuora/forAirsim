"""Read-only postflight audit, with saved-time checks isolated from controls."""
import sys,json,shutil,platform,importlib.metadata as metadata
from pathlib import Path
from collections import Counter
import numpy as np
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from analyze_step3_development import lines,csv_save,trajectory
from fixed_map_perception_task import live_estimate

def main(root):
    root=Path(root).resolve();trial=root/'online_trial';result=read(trial/'result.json');out=root/'audit';out.mkdir(exist_ok=False)
    metrics=[]
    for path in sorted(trial.glob('flight_*/timestamped.jsonl')):
        metric=trajectory(path.parent,out);ev=lines(path);ramps=[r['command_ramp_accel_m_s2'] for r in ev if r.get('kind')=='velocity_command_requested' and r.get('command_ramp_accel_m_s2') is not None]
        metric.update(command_ramp_max_m_s2=max(ramps,default=None),command_acceleration_interpretation='exact generator dt logged; not acceleration of physical aircraft',state_rpc_geometry_max_s=max([r.get('state_read_and_geometry_wall_s',0) for r in ev if r.get('kind')=='state'],default=None))
        metrics.append(metric)
    csv_save(out/'key_flight_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in metrics]);save(out/'flight_metrics.json',metrics)
    cfg=read(trial/'online/version_manifest.json')['config'];saved_cfg=dict(cfg,scene_epoch_required=False)
    frame_rows=[];indexes=[];selected=[];rejections=Counter();certificate_failures=[]
    for p in sorted((trial/'online/frames').glob('*/perception.json')):
        r=read(p);m=r['capture'];e=r['position_estimate'];valid_at_saved_time=False;reason=None
        if r['valid']:
            try:live_estimate(r,saved_cfg,r['processed_monotonic_s']);valid_at_saved_time=True
            except ValueError as ex:reason=str(ex);certificate_failures.append(dict(frame_id=r['frame_id'],reason=reason))
        rejections.update(r['rejection_reasons']);point=e.get('surface_world_xyz_m',[None]*3)
        frame_rows.append(dict(frame_id=r['frame_id'],candidate_count=len(r['semantic_frame']['candidates']),target_state=r['temporal_identity']['state'],track_id=r['temporal_identity']['track_id'],recorded_valid=r['valid'],saved_time_certificate=valid_at_saved_time,certificate_refusal=reason,inference_s=r['inference_wall_s'],capture_s=m['capture_wall_s'],receipt_to_processed_s=r['processed_monotonic_s']-m['received_monotonic_s'],rgb_timestamp_ns=m['rgb_timestamp_ns'],depth_timestamp_ns=m['depth_timestamp_ns'],pair_delta_ms=m['rgb_depth_delta_ms'],surface_x_m=point[0],surface_y_m=point[1],surface_z_m=point[2],refusals=';'.join(r['rejection_reasons'])))
        overlay=p.parent/('overlay.jpg' if (p.parent/'overlay.jpg').exists() else 'overlay.png')
        entry=dict(frame_id=r['frame_id'],capture=str(p.parent/'capture.json'),rgb=m['rgb_path'],depth=m['depth_path'],perception=str(p),overlay=str(overlay),rgb_sha256=m['image_sha256'],depth_sha256=m['depth_sha256'],target_generation=m['target_generation'],description_sha256=m['description_sha256'],scene_version=m['runtime_scene_version'],original_mode='online',audit_mode='saved original processing clock, not current flight input')
        indexes.append(entry)
        if valid_at_saved_time:selected.append(entry)
    csv_save(out/'frame_metrics.csv',frame_rows);(out/'frame_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in indexes),encoding='utf-8');save(out/'online_positive_samples.json',selected[-3:])
    # Include queue drops, capture errors and unfinished work, not only inferred
    # frames. Actual raw files and rejection events remain unmodified.
    sensor_events=lines(trial/'online/events.jsonl');by_frame={}
    for event in sensor_events:
        if event.get('frame_id'):by_frame.setdefault(event['frame_id'],[]).append(event)
    inferred={r['frame_id']:r for r in indexes};full=[]
    for folder in sorted((trial/'online/frames').glob('frame-*')):
        capture=folder/'capture.json';meta=read(capture) if capture.exists() else {}
        full.append(dict(frame_id=folder.name,status='processed' if folder.name in inferred else 'not_processed_or_capture_failed',files=[str(p) for p in folder.iterdir() if p.is_file()],capture=meta,processed_reference=inferred.get(folder.name),events=by_frame.get(folder.name,[])))
    (out/'all_frame_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in full),encoding='utf-8')
    save(out/'sensor_failure_events.json',[r for r in sensor_events if r['kind'] in ['CAPTURE_FAILED','INFERENCE_BLOCKED']])
    service_summary=read(trial/'online/service_summary.json')
    handoff=read(root/'freshness_root_cause.json');ground_fix=read(root/'fast_publication_ground_metrics.json')
    events=lines(trial/'session_events.jsonl');starts=[r for r in events if r['kind']=='PERCEPTION_START_ACCEPTED'];plans=[r for r in events if r['kind']=='PERCEPTION_PLAN_FRESH_REVALIDATED'];loss=[r for r in events if r['kind']=='PERCEPTION_OLD_ACTION_CANCELLED'];hovers=[r for r in events if r['kind']=='LANGUAGE_SENSOR_APPROACH_HOVER_PASSED']
    cpu_scope=dict(backend='frozen YOLO CPU full image 640; current RGB attributes plus local IoU continuity',true_instance_identity_claim=False,visible_surface_not_center=True,external_visual_model_requests=sum(r['kind']=='VLX_REQUEST' for r in lines(trial/'online/events.jsonl')),raw_exposure_synchronization_claim=False)
    basic=bool(result.get('perception_guidance_hover_passed')) and bool(selected) and bool(starts) and bool(plans) and all(r['landed'] and not r['api_control'] for r in result['final_ground'].values()) and all(not r['failures'] and r['max_continuous_motion_state_gap_s']<=.75 and r['max_speed_m_s']<=.8 and r['max_command_speed_m_s']<=.6500001 for r in metrics) and not certificate_failures
    decision=dict(utc=utc(),basic_closed_loop_passed=basic,online_frames=len(frame_rows),candidate_frames=sum(r['candidate_count']>0 for r in frame_rows),saved_time_valid_surface_frames=len(selected),certificate_failures=certificate_failures,perception_execution_starts=len(starts),fresh_revalidated_plans=len(plans),source_bound_perception_commands=sum(r['perception_command_count'] for r in metrics),arrival_hover_events=len(hovers),loss_stop_events=len(loss),final_ground=result['final_ground'],issue=result['issue'],refusals=dict(rejections),detector_scope=cpu_scope,random_position_multiple_trials='NOT_RUN',similar_target_ambiguity='SYNTHETIC_INTERFACE_CHECK_ONLY',occlusion_recovery='ONLINE_EVENTS_IF_PRESENT; NOT_A_PREREGISTERED_DISTURBANCE_TRIAL',four_method_pilot_ready=False)
    decision.update(raw_frame_directories=len(full),service_summary=service_summary,primary_blocker=handoff,secondary_blocker='Windows PermissionError on publication replace terminated inference during return; capture replace also failed on three frames',postflight_fixes=dict(decision_published_before_optional_archive=True,coordinator_period_s=.05,source_age_limit_unchanged_s=.5,final_pair_validation=True,bounded_windows_replace_retry_s=.1,ground_evidence=ground_fix,flight_retested=False),native_ui_check=dict(presentation=read(root/'qt_native_v9.json'),final_exit_code=0,earlier_exits=[-1073740791,-1073740791],cause='check script called executor-owned RPC client across threads; corrected to a separate main-thread probe'))
    save(root/'CLOSED_LOOP_DECISION.json',decision)
    archive=root/'delivery';archive.mkdir();files=[]
    modules=list((ROOT/'python').glob('fixed_map_*.py'))+[ROOT/'python/run_yolo_language_mission_v8.py',ROOT/'python/test_yolo_sensor_mission_v8.py',ROOT/'python/audit_yolo_closed_loop.py',ROOT/'python/check_yolo_dashboard_v9.py',ROOT/'scripts/unreal/build_random_region_v2.py']
    configs=[ROOT/'configs/fixed_map_perception_yolo_cpu_v9.json',ROOT/'configs/step3_balloon_cpu_dual_v9.json',ROOT/'configs/fixed_map_scene_runtime_v2.json',ROOT/'configs/fixed_map_scene_state.json',ROOT/'01_固定地图实验.cmd']
    for p in modules+configs:
        dest=archive/p.name;shutil.copyfile(p,dest);files.append(dict(source=str(p),snapshot=str(dest),sha256=sha(p)))
    maps=[]
    for name in ['EnclosedUAVLab_Manual.umap','EnclosedUAVLab_RuntimeRandom_v1.umap','EnclosedUAVLab_RuntimeRandom_v2.umap']:
        p=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual/Maps'/name;maps.append(dict(path=str(p),sha256=sha(p),copied=False))
    d=read(ROOT/'configs/step3_balloon_cpu_dual_v9.json');assert sha(d['weights'])==d['weights_sha256']
    save(archive/'manifest.json',dict(utc=utc(),python=platform.python_version(),packages={n:metadata.version(n) for n in ['torch','ultralytics','PyQt5','numpy','opencv-python']},files=files,maps=maps,weights=dict(path=d['weights'],sha256=d['weights_sha256'],copied=False),historical_references=['reports/fixed_map_flight_20261009','reports/step1_quick_review_20261010_083715','reports/step2_perception_20261010','reports/yolo_language_closed_loop_20261010_172830'],source_during_flight=str(root/'source_before_trial/manifest.json'),snapshot_verified=all(sha(r['snapshot'])==r['sha256'] for r in files)))
    table=[('自然语言→任务JSON','通过' if read(trial/'language.json').get('validated') else '未通过','本轮真实本地0.6B模型调用；结构化任务，不生成代码或坐标'),('真实采集','通过' if frame_rows else '未通过',f'{len(frame_rows)}帧；时间戳、RGB/深度响应位姿均归档；不声称曝光同步'),('YOLO检测','通过' if decision['candidate_frames'] else '未通过',f"{decision['candidate_frames']}帧有候选；同一冻结权重、conf=.25、CPU640"),('连续关联及表面定位','通过' if selected else '未通过',f'{len(selected)}帧通过原处理时刻复核；RGB属性＋局部连续性，不是实例真值或气球中心'),('新鲜规划与执行','通过' if plans and starts else '未通过',f'{len(plans)}个新鲜校验计划，{len(starts)}次执行；坐标来自本轮感知'),('感知接近后稳定悬停','通过' if basic else '未通过',f'{len(hovers)}次完整任务悬停事件；见实际轨迹和指标'),('返航落地及释放','通过' if all(v['landed'] and not v['api_control'] for v in result['final_ground'].values()) else '未通过','两机最终实际状态；DroneB地面待命'),('多位置、相似干扰、真实遮挡扰动','未完成','不把少量接口测试和单次闭环写成扰动收益')]
    text='# YOLO 自然语言闭环开发验收\n\n'+('本轮基础闭环通过；下一阶段仍需扰动验证。' if basic else '本轮基础闭环未通过；可以继续地面感知与规划开发。')+'\n\n|环节|判定|证据范围|\n|---|---|---|\n'+''.join(f'|{a}|{b}|{c}|\n' for a,b,c in table)+'\n本轮仅一次实际起飞：静态障碍依据的观察点[30,-9,3.1]、真实运动、观察悬停、返航落地。该段是观察准备，不是感知制导。没有有效在线感知计划被执行。已保存在线frame-000252的独立规划回放，接近点[38,-7.66745,3.1]，标注不可执行。\n\n阻塞：原处理时刻7帧有效，但发布额外耗时0.125–0.157秒；仅两帧抵达主进程时未过0.5秒期限，剩余不足0.063秒，原0.25秒任务检查可能错过。返航期间Windows文件替换PermissionError使推理进程退出；原记录和两帧误标有效但已过期的记录均保留。\n\n飞后修复：决策先发布、JPEG检测框、最后时效复核、任务检查0.05秒、文件替换最多100毫秒重试。地面复验24帧无目标、无时效拒绝，最大送达年龄0.391秒；37项接口测试和7项离线界面检查通过。文件重试有针对性单元检查。修复后的在线正例、规划执行、失锁停止和恢复均未复飞验证，不能记为通过。原生界面检查最初跨线程使用RPC客户端导致异常退出，修正检查脚本后退出码0；当前截图的检测框与路线明确为回放。\n\n历史地图与飞行记录保持原日期、版本；本轮在线、地面、回放和失败证据独立保存。气球在X42–49/Y−7–−1.5/Z2.4–5m内无系绳静态随机生成，飞行高度仍为3.1m，接近点限制X≤38m以避开未观测区域。YOLO仅检测气球类别，指定目标由当前RGB属性和局部连续关联筛选；不声称真实实例身份或气球中心定位。未查询目标真值作控制输入。\n\n交付证据：delivery/manifest.json为当前源码、配置、环境和大文件引用；source_before_trial/manifest.json为飞前版本；audit/all_frame_index.jsonl包含全部已保存帧和拒识事件；audit/key_flight_metrics.csv和轨迹/指令CSV为实际复算；online_trial为不可改写原始失败记录。CLOSED_LOOP_DECISION.json记录详细判定。\n\n下一次基础验证必须检查修复后新鲜正例→来源绑定规划→实际接近→稳定悬停→返航释放；基础通过后再做位置、相似干扰与真实遮挡试次。现阶段不具备四方法先导条件。\n'
    (root/'ACCEPTANCE.md').write_text(text,encoding='utf-8');print(json.dumps(decision,ensure_ascii=False))

if __name__=='__main__':main(sys.argv[1])

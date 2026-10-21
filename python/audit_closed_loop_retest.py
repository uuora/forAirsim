"""Postflight evidence audit only. No truth, motion or saved-frame promotion."""
import sys,json,shutil,platform,time
from pathlib import Path
from collections import Counter
from fixed_map_perception import read,save,sha,utc,ROOT
from fixed_map_perception_task import live_estimate
from analyze_step3_development import lines,csv_save,trajectory

def audit(root):
    root=Path(root).resolve();trial=root/'online_trial';out=root/'audit';out.mkdir(exist_ok=False)
    result=read(trial/'result.json');declaration=read(root/'RETEST_DECLARATION.json')
    cfg=read(trial/'online/version_manifest.json')['config'];offline_cfg=dict(cfg,scene_epoch_required=False)
    event_rows=lines(trial/'session_events.jsonl');sensor_events=lines(trial/'online/events.jsonl')
    kinds=Counter(r['kind'] for r in event_rows);reject=Counter();frames=[];samples=[];full=[];failures=[]
    grouped={}
    for r in sensor_events:
        if r.get('frame_id'):grouped.setdefault(r['frame_id'],[]).append(r)
    for folder in sorted((trial/'online/frames').glob('frame-*')):
        p=folder/'perception.json';capture=folder/'capture.json'
        index=dict(frame_id=folder.name,files=[str(f) for f in folder.iterdir() if f.is_file()],events=grouped.get(folder.name,[]),status='NO_INFERENCE_RECORD')
        if capture.exists():index['capture']=read(capture)
        if p.exists():
            r=read(p);m=r.get('capture',{});e=r.get('position_estimate',{});certificate=False;reason=None
            if r.get('valid'):
                try:live_estimate(r,offline_cfg,r['processed_monotonic_s']);certificate=True
                except (ValueError,KeyError,TypeError) as ex:reason=str(ex);failures.append(dict(frame_id=folder.name,reason=reason))
            reject.update(r.get('rejection_reasons',[]));point=e.get('surface_world_xyz_m',[None]*3)
            f=dict(frame_id=folder.name,recorded_valid=r.get('valid'),saved_original_processing_certificate=certificate,certificate_refusal=reason,candidate_count=len(r.get('semantic_frame',{}).get('candidates',[])),target_state=r.get('temporal_identity',{}).get('state'),local_track_id=r.get('temporal_identity',{}).get('track_id'),inference_s=r.get('inference_wall_s'),rgb_timestamp_ns=m.get('rgb_timestamp_ns'),depth_timestamp_ns=m.get('depth_timestamp_ns'),readback_pair_delta_ms=m.get('rgb_depth_delta_ms'),source_age_on_processing_s=r.get('processed_monotonic_s',m.get('received_monotonic_s',0))-m.get('received_monotonic_s',0),surface_x_m=point[0],surface_y_m=point[1],surface_z_m=point[2],rejection_reasons=';'.join(r.get('rejection_reasons',[])))
            frames.append(f);index.update(status='PROCESSED',perception_path=str(p),certificate_valid_at_original_processing=certificate)
            if certificate:samples.append(dict(frame_id=folder.name,perception=str(p),capture=str(capture),rgb=m['rgb_path'],depth=m['depth_path'],overlay=r.get('overlay_path',str(folder/'overlay.jpg')),rgb_sha256=m['image_sha256'],depth_sha256=m['depth_sha256'],mode='REAL_ONLINE_EVIDENCE_AT_ORIGINAL_TIME_NOT_CURRENT_CONTROL_INPUT'))
        full.append(index)
    csv_save(out/'frame_metrics.csv',frames)
    (out/'all_frame_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in full),encoding='utf-8')
    save(out/'online_sample_references.json',samples[-3:]);save(out/'recorded_valid_certificate_failures.json',failures)
    save(out/'sensor_worker_failures.json',[r for r in sensor_events if r['kind'] in ['CAPTURE_REJECTED','INFERENCE_BLOCKED']])
    metrics=[]
    for folder in sorted(trial.glob('flight_*')):
        metric=trajectory(folder,out);ev=lines(folder/'timestamped.jsonl');ramps=[r['command_ramp_accel_m_s2'] for r in ev if r.get('kind')=='velocity_command_requested' and r.get('command_ramp_accel_m_s2') is not None]
        metric.update(command_ramp_max_m_s2=max(ramps,default=None),command_acceleration_interpretation='exact logged generator dt; physical state acceleration and event-time jitter derivative remain separately listed',state_rpc_geometry_max_s=max([r.get('state_read_and_geometry_wall_s',0) for r in ev if r['kind']=='state'],default=None))
        metrics.append(metric)
    save(out/'flight_metrics.json',metrics);csv_save(out/'key_flight_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in metrics])
    confirmation=[r for r in event_rows if r['kind']=='LANGUAGE_SENSOR_CONFIRM_REFUSED'];csv_save(out/'confirmation_refusals.csv',[dict(utc=r['utc'],frame_id=r['refusal']['frame_id'],reason=r['refusal']['reason']) for r in confirmation])
    bound=[]
    for r in event_rows:
        if r['kind']=='PERCEPTION_MOTION_EVIDENCE':bound.append(dict(utc=r['utc'],plan_version=r['plan_version'],state_timestamp_ns=r['state_timestamp_ns'],**{k:v for k,v in r['binding'].items() if not isinstance(v,(dict,list))}))
    csv_save(out/'execution_source_binding.csv',bound)
    old_stops=[r for r in event_rows if r['kind']=='PERCEPTION_OLD_ACTION_CANCELLED'];save(out/'loss_stop_events.json',old_stops)
    # Runtime declarations and hash receipts are copied as small input evidence.
    archive=root/'delivery';archive.mkdir();current=[]
    before=read(root/'source_before_trial/manifest.json')
    for row in before['files']:
        p=Path(row['source']);dest=archive/p.name;shutil.copyfile(p,dest)
        current.append(dict(source=str(p),snapshot=str(dest),sha256=sha(p),flight_sha256=row['sha256'],unchanged_since_flight=sha(p)==row['sha256']))
    self_path=Path(__file__);shutil.copyfile(self_path,archive/self_path.name)
    save(archive/'manifest.json',dict(utc=utc(),files=current,auditor_sha256=sha(self_path),maps=before['maps'],weights=before['weights'],python=platform.python_version(),previous_evidence=declaration['previous'],no_large_asset_copies=True))
    final=result['final_ground'];ground_ok=set(final)=={'DroneA','DroneB'} and all(r.get('landed') is True and r.get('api_control') is False for r in final.values())
    if 'ground_verification_passed' in result:ground_ok=ground_ok and result['ground_verification_passed'] is True
    execution_count=kinds['PERCEPTION_START_ACCEPTED'];plan_count=kinds['PERCEPTION_PLAN_FRESH_REVALIDATED'];hovers=kinds['LANGUAGE_SENSOR_APPROACH_HOVER_PASSED'];bound_commands=sum(r['perception_command_count'] for r in metrics)
    safety_ok=bool(metrics) and all(not m['failures'] and m['max_continuous_motion_state_gap_s']<=.75 and m['max_speed_m_s']<=.8 and m['max_command_speed_m_s']<=.6500001 and m['minimum_approach_return_envelope_clearance_m']>=.75 and m['command_ramp_max_m_s2']<=.4000001 for m in metrics)
    basic=bool(result.get('perception_guidance_hover_passed')) and result.get('issue') is None and result['mission'].get('outcome')=='APPROACH_HOVER_PASSED' and bool(samples) and execution_count>0 and plan_count>0 and bound_commands>0 and ground_ok and safety_ok and not failures
    decision=dict(utc=utc(),basic_closed_loop_passed=basic,online_processed_frames=len(frames),raw_frame_directories=len(full),candidate_frames=sum(r['candidate_count']>0 for r in frames),valid_surface_frames_at_original_processing=len(samples),recorded_valid_certificate_failures=failures,planning_fresh_revalidation_count=plan_count,execution_start_count=execution_count,source_bound_actual_commands=bound_commands,stable_approach_hover_events=hovers,loss_stop_events=len(old_stops),recovery_plan_count=sum(r.get('actor')=='fresh_reconfirmation_replan' for r in event_rows if r['kind']=='PERCEPTION_PLAN_SUBMITTED'),issue=result['issue'],mission_outcome=result['mission'].get('outcome'),latest_refusal=result['mission'].get('last_refusal'),final_ground=final,safety_checks_passed=safety_ok,all_stage_max_state_gap_s=max([m['max_state_gap_s'] for m in metrics],default=None),continuous_motion_max_state_gap_s=max([m['max_continuous_motion_state_gap_s'] for m in metrics],default=None),refusals=dict(reject),vlx_requests=sum(r['kind']=='VLX_REQUEST' for r in sensor_events),random_position_trials='NOT_RUN',similar_distractor_trial='NOT_RUN',real_occlusion_trial='NOT_RUN',four_method_pilot_ready=False,no_formal_algorithm_benefit_claim=True)
    retention=result.get('raw_retention',{})
    decision['raw_record_retention']=retention
    decision['available_raw_rgb_depth_pairs']=sum((folder/'rgb.png').is_file() and (folder/'depth_ray_range_m.npy').is_file() for folder in (trial/'online/frames').glob('frame-*'))
    save(root/'CLOSED_LOOP_DECISION.json',decision)
    rows=[('自然语言任务',bool(read(trial/'language.json').get('validated')),'本轮实际本地模型调用，封闭JSON和目标版本绑定'),('在线采集与YOLO候选',bool(frames) and decision['candidate_frames']>0,f"{len(frames)}帧处理，{decision['candidate_frames']}帧候选"),('属性关联与深度表面定位',bool(samples),f'{len(samples)}帧原处理时刻证据有效；不是实例真值或气球中心'),('新鲜规划与实际感知执行',bool(plan_count and execution_count and bound_commands),f'{plan_count}次规划复核，{execution_count}次执行启动，{bound_commands}条来源绑定实际指令'),('任务接近与稳定悬停',basic,f'{hovers}次感知接近悬停事件；完整判据及安全复算须同时通过'),('返航落地释放',ground_ok,'两架机落地，API控制释放'),('失锁停止与恢复',False,f'{len(old_stops)}次自然失效停止；不等同冻结遮挡扰动试次')]
    text='# 闭环针对性复验\n\n'+('基础闭环通过，扰动仍待验证。' if basic else '基础闭环未通过；本轮不重复起飞，保留具体阻塞。')+'\n\n|环节|判定|实际证据|\n|---|---|---|\n'+''.join(f"|{a}|{'通过' if b else '未通过/未完成'}|{c}|\n" for a,b,c in rows)
    text+='\n实际任务结果：'+str(decision['mission_outcome'])+'；驱动问题：'+str(result['issue'])+'；最后确认拒识：'+str(decision['latest_refusal'])+'。全部拒识与中断保留。\n\n观察准备使用事先声明的静态走廊点[30,-9,3.1]，不由目标Actor反推，不计感知制导成功。未重训、未调用VLX、未传送/reset无人机、未修改冻结Manual地图或底座参数。本轮沿用同一随机场景，未用旧固定坐标补定位。\n\n'
    text+='控制安全指标、实际轨迹和指令见audit/key_flight_metrics.csv及相邻CSV。全阶段最大状态间隔和连续运动间隔分别报告；指令斜坡精确dt、物理状态差分加速度、发送事件差分分别保留。所有表面估计指可见表面。工作区X≤38.5、接近点X≤38，未观察随机区域物体由保守距离限制，预览净空不证明全场安全。\n\n'
    text+='证据索引：audit/all_frame_index.jsonl涵盖未推理帧和原始采集元数据、位姿、拒绝事件；audit/execution_source_binding.csv逐状态绑定计划与感知证据；online_trial保留原始状态、指令、计划和事件；source_before_trial是飞前冻结；delivery/manifest.json列当前差异与大型地图/权重引用。\n\n'
    if retention:
        text+='原始载荷保留策略：落地及感知退出后保留全部原处理时刻有效帧、控制/计划引用帧和选定边界帧，清理未选中的原始RGB、深度与标注图。实际状态 '+str(retention.get('status'))+'，清理 '+str(retention.get('purged_bytes',0))+' 字节。删除清单与哈希见online_trial/RAW_RETENTION_MANIFEST.jsonl及RAW_RETENTION_RECEIPT.json；已清理帧只能查看元数据，不能重放完整原始视频或重跑全帧视觉分析。有效证书仍按原有RGB/深度哈希核验，不放宽门槛。\n\n'
    else:
        text+='本驱动未声明自动原始载荷清理；原始载荷是否仍在以实际文件和后续清理回执为准。\n\n'
    text+='历史上一轮失败的结果记录保留；后续原始载荷清理见各目录RAW_RECORD_RETENTION回执。扰动A位置变化、B相似干扰、C真实遮挡均未运行。未具备四方法先导条件；不能把少量开发试次表述为正式算法收益。\n'
    (root/'ACCEPTANCE.md').write_text(text,encoding='utf-8');print(json.dumps(decision,ensure_ascii=False))

if __name__=='__main__':audit(sys.argv[1])

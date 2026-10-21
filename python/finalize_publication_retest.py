"""Current revision and postflight ground receipt; no new mission."""
import sys,json,shutil,statistics
from pathlib import Path
from dataclasses import replace
from collections import Counter
from fixed_map_perception import ROOT,read,save,sha,utc
from fixed_map_yolo_identity import select
from t1_delayed_semantic_bridge import _frame_from_dict

root=Path(sys.argv[1]).resolve();out=root/'delivery_after_ground_fix';out.mkdir(exist_ok=False)
comparisons=[]
for path in sorted((root/'online_trial/online/frames').glob('*/perception.json')):
    r=read(path);frame=replace(_frame_from_dict(r['semantic_frame']),encoded_image=Path(r['capture']['rgb_path']).read_bytes())
    selected,audit=select(frame,'带白色横条的红色气球')
    old=r['current_attribute_audit']
    old_matches=old.get('matching_candidate_ids',old.get('attribute_details',{}).get('matching_candidate_ids',[]))
    new_matches=audit.get('matching_candidate_ids',audit.get('attribute_details',{}).get('matching_candidate_ids',[]))
    comparisons.append(dict(frame_id=frame.frame_id,same_matching_candidates=old_matches==new_matches,same_acceptance=(old['status']=='ACCEPTED')==(audit['status']=='ACCEPTED'),mode='OFFLINE_ATTRIBUTE_REPLAY_NOT_CONTROL',selected_candidate_id=selected))
assert all(r['same_matching_candidates'] and r['same_acceptance'] for r in comparisons),'ATTRIBUTE_REPLAY_CHANGED'
save(root/'OFFLINE_ATTRIBUTE_EQUIVALENCE.json',dict(scope='same saved RGB/YOLO boxes, unchanged RGB thresholds, no online freshness claim',frames=len(comparisons),all_match=True,records=comparisons))
label='online_ground_cpu_v9_decode_reuse';ground=read(root/(label+'_result.json'));rows=[read(p) for p in (root/label/'frames').glob('*/perception.json')]
pub=read(root/(label+'_publication_metrics.json'));ages=[r['source_age_on_parent_receipt_s'] for r in pub]
timing=dict(utc=utc(),mode='REAL_CURRENT_GROUND_NO_TARGET_NO_FLIGHT',frames=len(rows),ground_result=ground,max_parent_source_age_s=max(ages),parent_age_over_half_second=sum(a>.5 for a in ages),median_parent_age_s=statistics.median(ages),medians={k:statistics.median(r['processing_timings'][k] for r in rows) for k in ['archive_to_inference_s','receipt_to_capture_archive_s','attribute_stage_s','depth_and_decision_stage_s']},flight_retested_after_v10=False)
save(root/'POSTFLIGHT_GROUND_METRICS.json',timing)
paths=list((ROOT/'python').glob('fixed_map_*.py'))+[ROOT/'python/run_yolo_language_mission_v8.py',ROOT/'python/check_ground_yolo_cpu_v9.py',ROOT/'python/check_yolo_dashboard_v9.py',Path(__file__),ROOT/'configs/fixed_map_perception_yolo_cpu_v10.json',ROOT/'configs/step3_balloon_cpu_dual_v9.json',ROOT/'configs/fixed_map_scene_runtime_v2.json',ROOT/'configs/fixed_map_scene_state.json',ROOT/'configs/ground_replay_preview_reference.json',ROOT/'01_固定地图实验.cmd']
before={r['source']:r for r in read(root/'source_before_trial/manifest.json')['files']};files=[]
for p in paths:
    dest=out/p.name;shutil.copyfile(p,dest);previous=before.get(str(p),{})
    files.append(dict(source=str(p),snapshot=str(dest),sha256=sha(p),flight_sha256=previous.get('sha256'),changed_since_flight=sha(p)!=previous.get('sha256')))
from importlib.metadata import version
save(out/'manifest.json',dict(utc=utc(),files=files,packages={n:version(n) for n in ['torch','ultralytics','PyQt5','numpy','opencv-python']},large_asset_reference=str(root/'source_before_trial/manifest.json'),snapshot_verified=all(sha(r['snapshot'])==r['sha256'] for r in files),flight_verified=False))
import airsim
c=airsim.MultirotorClient(port=41454,timeout_value=3);final={}
for name in ['DroneA','DroneB']:
    a=c.getMultirotorState(vehicle_name=name)
    final[name]=dict(landed=a.landed_state==airsim.LandedState.Landed,api_control=c.isApiControlEnabled(vehicle_name=name),timestamp_ns=a.timestamp)
assert all(r['landed'] and not r['api_control'] for r in final.values())
save(root/'final_ground_receipt.json',dict(utc=utc(),ground=final,read_only=True))
current=read(root/'CLOSED_LOOP_DECISION.json');current['postflight_v10_ground']=timing;current['v10_flight_validated']=False;current['v9_publication_failure_reproduced']=bool(read(root/'audit/sensor_worker_failures.json'));save(root/'CLOSED_LOOP_DECISION.json',current)
path=root/'ACCEPTANCE.md'
with path.open('a',encoding='utf-8') as f:
    f.write('\n飞后针对性修复v10：同一不可变RGB字节仅解码一次，YOLO、属性校验和检测框共用；属性唯一性判定只算一次；捕获元数据仅归档一次。权重、属性阈值、相机尺寸、深度规则、时效.5秒和控制安全判据不变。'+str(len(comparisons))+'帧属性回放比较全部一致，回放不能进入在线控制。39项接口检查通过；此前误选测试文件的0项结果不计入验收。\n\n')
    f.write(f"修复后真实地面{len(rows)}帧无目标，无时效拒绝、无文件发布失败；送达主进程最大年龄{max(ages):.3f}秒。当前地面样本没有定位正例，不能用此结果宣称飞行闭环通过。具体耗时POSTFLIGHT_GROUND_METRICS.json；飞行中原处理耗时分解OFFLINE_PROCESSING_PROFILE.json。\n\n")
    f.write('飞行版本保存在source_before_trial和delivery，地面修复版本独立保存在delivery_after_ground_fix/manifest.json；原始本次失败和上一轮失败均未改写。下一次需验证v10在观察悬停期间稳定提供新鲜指定目标定位，并完成来源绑定规划、实际接近和回收。未放宽期限、未扩大场景、未再次起飞。\n')
metrics=read(root/'audit/flight_metrics.json')[0]
summary='# 版本、差异与证据引用\n\n本轮先用上一轮修复后的CPU v9做一次实际复验，再仅做v10地面修复；两版证据分开。\n\n'
summary+='飞前冻结：source_before_trial/manifest.json；本次原始失败：online_trial；当前小文件快照、哈希及环境：delivery_after_ground_fix/manifest.json。大型地图和YOLO权重沿用飞前引用与哈希，不重复复制。\n\n'
summary+='原Manual地图未改写；RuntimeRandom_v2当前种子与场景版本未改动，公开配置不含目标坐标；目标真值不进入感知或控制。YOLO权重不变、CPU640、conf=.25、NMS=.5。自然语言由现有本地模型生成闭合JSON，指定目标仍为带白色横条的红色气球。VLX调用0。\n\n|文件|飞后变化|影响|\n|---|---|---|\n'
for r in files:
    if r['changed_since_flight']:summary+='|'+Path(r['source']).name+'|哈希见清单|处理速度、记录/界面或版本绑定；尚无v10飞行证据|\n'
summary+=f"\n实际轨迹{metrics['actual_trajectory_length_m']:.3f}m，最大实测速度{metrics['max_speed_m_s']:.3f}m/s，指令上限{metrics['max_command_speed_m_s']:.3f}m/s，运动阶段最小净空{metrics['minimum_approach_return_envelope_clearance_m']:.3f}m；连续运动最大状态间隔{metrics['max_continuous_motion_state_gap_s']:.3f}s，全阶段最大间隔{metrics['max_state_gap_s']:.3f}s单列。精确生成器dt的斜坡上限{metrics['command_ramp_max_m_s2']:.3f}m/s²，物理状态加速度和指令发送抖动差分单独列CSV。无感知目标指令实际执行。\n\n"
summary+='历史引用：reports/fixed_map_flight_20261009、step1_quick_review_20261010_083715、step2_perception_20261010、closed_loop_cpu_recheck_20261011_102452。各原始记录保留原日期版本。当前判定见ACCEPTANCE.md；轨迹、指令、关键指标及全部帧见audit/；地面修复记录online_ground_cpu_v9_decode_reuse；属性重放OFFLINE_ATTRIBUTE_EQUIVALENCE.json。最后两机地面释放控制见final_ground_receipt.json。\n'
(root/'VERSION_AND_EVIDENCE.md').write_text(summary,encoding='utf-8');print(json.dumps(timing,ensure_ascii=False))

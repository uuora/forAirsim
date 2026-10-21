"""Audit surface/recovery development replay; no aircraft commands."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from fixed_map_perception import ROOT,read,save,sha,utc
from fixed_map_perception_geometry import surface_from_box
from fixed_map_perception_execution import static_geometry_v2
from fixed_map_mission import Geometry
from analyze_step3_development import csv_save,lines

root=Path(sys.argv[1]).resolve();cfg=read(ROOT/'configs/fixed_map_perception_yolo_cpu_v11.json');rows=[];previous=None;old_jumps=0;new_jumps=0
for p in sorted((root/'online_trial/online/frames').glob('*/perception.json')):
    r=read(p)
    if not r['valid']:continue
    selected=r['temporal_identity']['selected_candidate_id'];c=next(c for c in r['semantic_frame']['candidates'] if c['candidate_id']==selected);m=r['capture']
    new=surface_from_box(c['bbox_xyxy_pixels'],np.load(m['depth_path']),m,cfg,r['processed_monotonic_s']);old=r['position_estimate']['surface_world_xyz_m']
    a=b=None
    if previous and m['frame_sequence']==previous[0]+1:
        a=float(np.linalg.norm(np.array(old)-previous[1]));b=float(np.linalg.norm(np.array(new['surface_world_xyz_m'])-previous[2]));old_jumps+=a>.25;new_jumps+=b>.25
    rows.append(dict(frame_id=r['frame_id'],source_sha256=m['image_sha256'],source_mode='ORIGINAL_ONLINE_RAW_REPROCESSED_OFFLINE_NOT_CONTROL',old_x=old[0],old_y=old[1],old_z=old[2],new_x=new['surface_world_xyz_m'][0],new_y=new['surface_world_xyz_m'][1],new_z=new['surface_world_xyz_m'][2],old_adjacent_jump_m=a,new_adjacent_jump_m=b,actual_depth_pixel_x=new['representative_depth_pixel_xy'][0],actual_depth_pixel_y=new['representative_depth_pixel_xy'][1],actual_depth_m=new['ray_range_m'],depth_band_m=new['selection']['depth_band_m'],center_estimated=new['center_estimated']))
    previous=(m['frame_sequence'],np.array(old),np.array(new['surface_world_xyz_m']))
csv_save(root/'audit/surface_policy_DEVELOPMENT_REPLAY.csv',rows)
events=lines(root/'online_trial/session_events.jsonl');rejected=next(e for e in events if e['kind']=='PERCEPTION_PLAN_REJECTED');states=[e for p in (root/'online_trial').glob('flight_*/timestamped.jsonl') for e in lines(p) if e['kind']=='state'];state=min(states,key=lambda s:abs(s['host_monotonic_s']-rejected['host_monotonic_s']))
parts,_=static_geometry_v2(read(ROOT/'reports/fixed_map_flight_20261009/fixed_version/editor_audit/planning_geometry.json')['primitives']);g=Geometry(parts,.75);point=state['position_xyz_m'];plane=[point[0],point[1],3.1];column=g.segment(point,plane,1.25)
replay=dict(mode='DEVELOPMENT_REPLAY_NOT_PROSPECTIVE_FLIGHT',frames=len(rows),old_adjacent_surface_jumps_over_025m=old_jumps,new_adjacent_surface_jumps_over_025m=new_jumps,source_point_meaning='actual pixel visible surface NOT center',same_geometry_and_source_images=True,engineering_depth_band_not_calibrated=True,failure_point_xyz_m=point,original_height_error_m=abs(point[2]-3.1),unchanged_plan_height_guard_m=.03,recovery_column=dict(from_xyz_m=point,to_xyz_m=plane,clearance_required_m=1.25,passed=column[0],receipt=column[1]),return_alignment_controller_executed_in_trial=any(e['kind']=='RETURN_HEIGHT_ALIGNMENT' for e in events),flight_validated_after_fix=False)
save(root/'V11_DEVELOPMENT_REPLAY.json',replay)
label='online_ground_cpu_v9_surface_recovery_v11';ground=read(root/(label+'_result.json'));pub=read(root/(label+'_publication_metrics.json'));ages=[e['source_age_on_parent_receipt_s'] for e in pub];ground['max_parent_age_s']=max(ages);save(root/'V11_GROUND_CHECK.json',ground)
archive=root/'delivery_v11_ground_only';archive.mkdir(exist_ok=False);before={e['source']:e for e in read(root/'source_before_trial/manifest.json')['files']};files=[]
paths=list((ROOT/'python').glob('fixed_map_*.py'))+[ROOT/'python/run_yolo_language_mission_v8.py',ROOT/'python/prepare_closed_loop_retest.py',ROOT/'python/check_yolo_dashboard_v9.py',Path(__file__),ROOT/'configs/fixed_map_perception_yolo_cpu_v11.json',ROOT/'configs/step3_balloon_cpu_dual_v9.json',ROOT/'configs/fixed_map_scene_state.json',ROOT/'configs/fixed_map_scene_runtime_v2.json',ROOT/'01_固定地图实验.cmd']
for p in paths:
    dest=archive/p.name;shutil.copyfile(p,dest);files.append(dict(source=str(p),snapshot=str(dest),sha256=sha(p),flight_sha256=before.get(str(p),{}).get('sha256'),changed_since_flight=sha(p)!=before.get(str(p),{}).get('sha256')))
save(archive/'manifest.json',dict(utc=utc(),files=files,large_files_reference=str(root/'source_before_trial/manifest.json'),snapshot_verified=all(sha(e['snapshot'])==e['sha256'] for e in files),flight_verified=False,prior_evidence='reports/closed_loop_publication_retest_20261011_123450'))
decision=read(root/'CLOSED_LOOP_DECISION.json');decision.update(v11_ground=ground,v11_development_replay=replay,v11_flight_validated=False);save(root/'CLOSED_LOOP_DECISION.json',decision)
metrics=read(root/'audit/flight_metrics.json')[0];stops=[s for s in metrics['stop_response'] if 'PERCEPTION_EVIDENCE_INVALID' in s['reason']]
with (root/'ACCEPTANCE.md').open('a',encoding='utf-8') as f:
    f.write('\n本次v10有实际感知计划和6条来源绑定控制指令，111帧原时刻定位有效；表面点变化触发停止，原停止响应'+str(stops[0]['first_below'])+'秒、后续最大漂移'+str(stops[0]['maximum_drift'])+'米。不是完整接近到达成功。\n\n')
    f.write('v11独立修复：稳健深度带内选取靠近RGB框中心的真实深度像素，仍为可见表面而非气球中心；带宽max(.04m,3×MAD)是工程采样规则，不是校准误差界。恢复先复用本次实际返航已执行的定高控制器，再读取新鲜证据并重新规划；过期时继续悬停等待原任务期限，废弃失效缓存路径。原0.25m表面稳定阈值、.5s时效、.03m规划高度检查、3.1m飞行高度和底座判据全部保留。\n\n')
    f.write(f'46项接口检查通过，v11真实地面{ground["processed_frames"]}帧无目标，拒识按原规则保留；最大送达年龄{max(ages):.3f}秒。{len(rows)}帧开发回放中相邻表面跳变超过.25m由{old_jumps}次变为{new_jumps}次；该数据参与本次开发，不是独立验证或正式算法收益。回放恢复定高列净空检查'+str(column[0])+'，不能替代恢复飞行。\n\n')
    f.write('本轮仅一次实际起飞，v11未再复飞；仍需验证修复后实际接近、稳定悬停、失锁停止后恢复及落地释放，随后才验证A/B/C扰动。飞行快照source_before_trial及delivery、地面修复快照delivery_v11_ground_only互相独立；旧失败、原始数据均未改写。\n')
text='# 版本与证据引用\n\n飞行版本CPU v10，当前界面默认CPU v11地面配置，禁止自动起飞。冻结地图和权重引用source_before_trial/manifest.json，不重复复制。当前源码、配置及飞前差异见delivery_v11_ground_only/manifest.json；环境原数据见online_trial/online/version_manifest.json；原Safety/task及settings快照保留在source_before_trial。\n\n'
text+='本轮保持随机场景版本与种子，未查询目标真值作为控制输入；无VLX、无重训、无reset或无人机传送。来源绑定计划、逐状态实际指令见audit/execution_source_binding.csv、相邻commands.csv及online_trial/perception_plan记录。全部帧audit/all_frame_index.jsonl；原时刻在线正例audit/online_sample_references.json。v11点选择变化与列检查V11_DEVELOPMENT_REPLAY.json及surface_policy_DEVELOPMENT_REPLAY.csv，明确离线开发回放；地面新采集V11_GROUND_CHECK.json。\n\n'
text+=f"实际轨迹{metrics['actual_trajectory_length_m']:.3f}m，最大速度{metrics['max_speed_m_s']:.3f}m/s，连续运动最大状态间隔{metrics['max_continuous_motion_state_gap_s']:.3f}s，全日志最大间隔{metrics['max_state_gap_s']:.3f}s单列；最大指令斜坡{metrics['command_ramp_max_m_s2']:.3f}m/s²，运动最小包络净空{metrics['minimum_approach_return_envelope_clearance_m']:.3f}m。底座和停止判据未修改，原几何52项物理障碍继续引用，未建模物体仍采用X38.5工作区、X38接近点及3.5m最小距离限制。\n\n"
text+='历史引用：reports/fixed_map_flight_20261009、step1_quick_review_20261010_083715、step2_perception_20261010、closed_loop_cpu_recheck_20261011_102452、closed_loop_publication_retest_20261011_123450。各原日期版本保留。最终判定见ACCEPTANCE.md及CLOSED_LOOP_DECISION.json；尚未具备四方法先导条件。\n'
(root/'VERSION_AND_EVIDENCE.md').write_text(text,encoding='utf-8')
import airsim
c=airsim.MultirotorClient(port=41454,timeout_value=3);final={n:dict(landed=c.getMultirotorState(vehicle_name=n).landed_state==airsim.LandedState.Landed,api_control=c.isApiControlEnabled(vehicle_name=n)) for n in ['DroneA','DroneB']};assert all(s['landed'] and not s['api_control'] for s in final.values());save(root/'final_ground_receipt.json',dict(utc=utc(),ground=final))
print(json.dumps(dict(replay=replay,ground=ground),ensure_ascii=False))

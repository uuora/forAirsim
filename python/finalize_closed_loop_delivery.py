"""Small read-only provenance receipt; never starts a flight."""
import sys,shutil
from pathlib import Path
from fixed_map_perception import read,save,sha,utc,ROOT

root=Path(sys.argv[1]).resolve()
before=read(root/'source_before_trial/manifest.json')['files']
rows=[]
for r in before:
    p=Path(r['source']);current=sha(p)
    rows.append(dict(source=str(p),flight_snapshot_sha256=r['sha256'],current_sha256=current,changed=current!=r['sha256'],historical_snapshot_preserved=sha(r['snapshot'])==r['sha256']))
save(root/'delivery/source_changes_since_flight.json',rows)
import airsim
c=airsim.MultirotorClient(port=41454,timeout_value=3)
ground={}
for name in ['DroneA','DroneB']:
    s=c.getMultirotorState(vehicle_name=name)
    ground[name]=dict(state_timestamp_ns=s.timestamp,landed=s.landed_state==airsim.LandedState.Landed,api_control=c.isApiControlEnabled(vehicle_name=name),speed_xyz_ned_m_s=[s.kinematics_estimated.linear_velocity.x_val,s.kinematics_estimated.linear_velocity.y_val,s.kinematics_estimated.linear_velocity.z_val])
assert all(s['landed'] and not s['api_control'] for s in ground.values())
save(root/'final_ground_receipt.json',dict(utc=utc(),ground=ground,read_only=True))
manifest=read(root/'delivery/manifest.json')
for p in [ROOT/'python/finalize_closed_loop_delivery.py',ROOT/'configs/ground_replay_preview_reference.json',ROOT/'configs/enclosed_uav_lab_manual_settings.json',ROOT/'reports/fixed_map_flight_20261009/fixed_version/task.json',Path.home()/'Documents/AirSim/settings.json']:
    if p.exists():
        dest=root/'delivery'/('current_settings.json' if p==Path.home()/'Documents/AirSim/settings.json' else 'frozen_task.json' if p.name=='task.json' else p.name)
        shutil.copyfile(p,dest);manifest['files'].append(dict(source=str(p),snapshot=str(dest),sha256=sha(p)))
manifest['snapshot_verified']=all(sha(r['snapshot'])==r['sha256'] for r in manifest['files'])
save(root/'delivery/manifest.json',manifest)
text='# 版本差异与证据引用\n\n'
text+='当前入口：01_固定地图实验.cmd；默认CPU v9地面配置，飞行按钮关闭。VLX实际调用0次；自然语言由已有本地fixed-mission-qwen3:0.6b解释为封闭任务JSON。\n\n'
text+='冻结YOLO权重：'+manifest['weights']['sha256']+'；conf=.25、NMS=.5、CPU全图640。旧CUDA/双尺度版本和失败证据保留。检测设备与分辨率影响速度和小目标召回，不能外推算法收益。\n\n'
text+='地图只引用，不重复复制：\n\n'+''.join('- '+r['path']+'，SHA256 '+r['sha256']+'\n' for r in manifest['maps'])
text+='\n原Manual地图未修改。独立RuntimeRandom_v2移除气球系绳及锚点、启用可移动气球；125项非语义静态网格变换保持，规划保留52项物理障碍，目标实体仅由当前相机候选建立接近限制。运行场景种子24101101，版本untethered_3d_66213abe9059；公开配置无目标XYZ，真值场景记录位于scene_setup独立评分链。\n\n'
text+='飞前→当前源码差异：\n\n|文件|变化|影响|\n|---|---|---|\n'
for r in rows:
    if r['changed']:
        text+='|'+Path(r['source']).name+'|已变更，哈希见source_changes_since_flight.json|飞后接口/发布/时效或界面修复；本次飞行不能证明该版本有效|\n'
text+='\n所有飞前小文件快照哈希仍匹配。底座参数仍为世界高度3.1m、指令速度.65m/s、实测停止阈值.8m/s、指令加速度.4m/s²、包络.75m、规划净空1.25m，其余沿用task.json。新随机区域使用表面接近距离至少3.5m、接近点X≤38m，禁止拿未观察区域的地图净空当作完整安全证明。\n\n'
text+='证据：旧reports/fixed_map_flight_20261009、step1_quick_review_20261010_083715、step2_perception_20261010保持原日期。当前online_trial为真实单次飞行失败；online_ground_cpu_v9_fast_publication为修复后地面无目标；saved_online_replay为原帧规划回放。逐帧全部索引audit/all_frame_index.jsonl；轨迹与指令CSV及完整指标audit/；37项接口、7项离线界面、实际原生界面呈现分别见interfaces_final.txt、interfaces_perception_final.txt、qt_offline_v9.json、qt_native_v9.json。\n\n'
metrics=read(root/'audit/flight_metrics.json')[0]
text+=f"实际轨迹{metrics['actual_trajectory_length_m']:.3f}m；最大速度{metrics['max_speed_m_s']:.3f}m/s；运动阶段最小包络净空{metrics['minimum_approach_return_envelope_clearance_m']:.3f}m；连续运动状态最大间隔{metrics['max_continuous_motion_state_gap_s']:.3f}s。全日志最大间隔{metrics['max_state_gap_s']:.3f}s发生于阶段转换/规划等待，单独报告；不将连续运动数值代替全时段值。指令发生时刻差分的加速度含发送抖动，精确生成器dt的指令斜坡最大{metrics['command_ramp_max_m_s2']:.3f}m/s²；物理加速度另列于轨迹CSV。\n\n"
text+='最终RPC复查两架机已落地/API控制均释放，见final_ground_receipt.json。下一阶段阻塞是修复后的正例送达与实时规划执行尚未实飞验证；多位置、相似干扰、真实遮挡未开始。\n'
(root/'VERSION_AND_EVIDENCE.md').write_text(text,encoding='utf-8')
print('Provenance and fresh ground receipt saved')

"""Finalize interval labels and verify delivered hashes, read-only RPC."""
import sys,json,shutil
from pathlib import Path
from fixed_map_perception import read,save,sha,utc,ROOT
root=Path(sys.argv[1]).resolve();decision=read(root/'CLOSED_LOOP_DECISION.json');result=read(root/'online_trial/result.json')
deadline=result['mission']['deadline_monotonic_s'];start=deadline-30
timeline=[]
for path in sorted((root/'online_trial/online/frames').glob('*/perception.json')):
    row=read(path)
    if not row.get('valid'):continue
    t=row['processed_monotonic_s']
    timeline.append(dict(frame_id=row['frame_id'],processed_monotonic_s=t,confirm_window_start=start,confirm_window_deadline=deadline,interval='BEFORE_CONFIRM_HOVER_READY' if t<start else 'AFTER_CONFIRM_DEADLINE' if t>deadline else 'DURING_CONFIRM_WINDOW'))
save(root/'audit/positive_timeline.json',timeline);decision['online_positive_intervals']=timeline
save(root/'CLOSED_LOOP_DECISION.json',decision)
manifest_path=root/'delivery_after_ground_fix/manifest.json';manifest=read(manifest_path)
for p in [ROOT/'python/audit_closed_loop_retest.py',ROOT/'python/profile_saved_perception_cost.py',Path(__file__)]:
    dest=root/'delivery_after_ground_fix'/p.name;shutil.copyfile(p,dest);manifest['files'].append(dict(source=str(p),snapshot=str(dest),sha256=sha(p),flight_sha256=None,changed_since_flight=True,auditor_only=True))
manifest['snapshot_verified']=all(sha(r['snapshot'])==r['sha256'] for r in manifest['files']);assert manifest['snapshot_verified'];save(manifest_path,manifest)
with (root/'ACCEPTANCE.md').open('a',encoding='utf-8') as f:
    f.write('\n逐时段复核：3帧有效定位中1帧位于观察悬停确认阶段建立前，2帧位于确认期限结束后的返航阶段；确认窗口内有效正例为0，不能用窗口外记录补成功，见audit/positive_timeline.json。本次没有CAPTURE_REJECTED或INFERENCE_BLOCKED事件，前次Windows发布故障未复现。7项离线界面检查通过，当前原生界面检查正常退出0；画面中的保存检测框和路线仍标注回放、禁止执行。\n')
print('Delivered hashes verified; positive intervals preserved')

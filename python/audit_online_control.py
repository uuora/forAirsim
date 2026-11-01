"""Read-only audit of protocol16 states, C2 windows, hits and resource counts."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from dual_target_control.online_curve import ReferenceTimeline
from dual_target_control.online_results import read_lines
from dual_target_control.reproduction_paths import workspace_record_path,leaf_name


def audit_run(folder):
    folder=Path(folder).resolve();rows=read_lines(folder/'online_states.jsonl');events=read_lines(folder/'online_events.jsonl');journal=read_lines(folder/'online_worker_evaluations.jsonl')
    windows=json.loads((folder/'online_reference_windows.json').read_text(encoding='utf-8'))
    m=json.loads((folder/'online_metrics.json').read_text(encoding='utf-8'));manifest=json.loads((folder/'online_manifest.json').read_text(encoding='utf-8'))
    if not rows or not windows:raise ValueError('记录不完整，没有可重算状态或参考')
    timeline=ReferenceTimeline(windows[0]['request'],windows[0]['parameters'])
    for w in windows[1:]:timeline.add(w['start_s'],w['request'],w['parameters'],w['source'],now=w['start_s']-.1)
    t=np.array([r['reference_elapsed_s'] for r in rows]);p=np.array([r['position_zup_m'] for r in rows]);ref=np.array([timeline.at(float(s))[0][0] for s in t]);err=np.linalg.norm(p-ref,axis=1)
    rmse=float(np.sqrt(np.trapz(err**2,t)/(t[-1]-t[0])))
    search_calls=sum(j['kind']=='EVALUATION_COMPLETE' for j in journal);validations=sum(j['kind']=='VALIDATION_COMPLETE' for j in journal)
    hits=[e for e in events if e['stage']=='TARGET_REACHED'];gates=[];goals=manifest['conditions']['online']['targets']
    for event in hits:
        i=int(np.searchsorted(t,event['reference_time_s']));a=rows[max(0,i-1)];b=rows[min(len(rows)-1,i)]
        goal=np.array(goals[event['target_index']]);p0=np.array(a['position_zup_m']);step=np.array(b['position_zup_m'])-p0
        aa=float(step@step);bb=2*float((p0-goal)@step);cc=float((p0-goal)@(p0-goal))-.25
        alpha=max(0.,min(1.,(-bb-math.sqrt(max(0.,bb*bb-4*aa*cc)))/(2*aa))) if cc>0 and aa>1e-12 else 0.
        point=p0+alpha*step;delta=goal-point
        yaw=a['yaw_rad']+alpha*math.atan2(math.sin(b['yaw_rad']-a['yaw_rad']),math.cos(b['yaw_rad']-a['yaw_rad']))
        angle=abs(math.degrees(math.atan2(math.sin(math.atan2(delta[1],delta[0])-yaw),math.cos(math.atan2(delta[1],delta[0])-yaw))))
        gates.append(dict(target_index=event['target_index'],entry_distance_m=float(np.linalg.norm(point-goal)),yaw_error_deg=angle,passed=bool(np.linalg.norm(point-goal)<=.500001 and angle<=20.000001)))
    files=('online_manifest.json','online_metrics.json','online_states.jsonl','online_events.jsonl','online_reference_windows.json','online_worker_evaluations.jsonl')
    source_ok=all((folder/'source_snapshot'/leaf_name(path)).is_file() and hashlib.sha256((folder/'source_snapshot'/leaf_name(path)).read_bytes()).hexdigest()==digest for path,digest in manifest['source_sha256'].items())
    decoder_ok=all(workspace_record_path(path).is_file() and hashlib.sha256(workspace_record_path(path).read_bytes()).hexdigest()==digest for path,digest in manifest['source_sha256'].items() if leaf_name(path) in ('online_curve.py','curves.py','parameterizations.py','route_curve.py','curve_task.py'))
    result=dict(run_dir=str(folder),condition_fingerprint=manifest['condition_fingerprint'],outcome=m['outcome'],
        rmse_difference=abs(rmse-m['position_rmse_m']),peak_difference=abs(float(err.max())-m['max_position_error_m']),
        worker_counts_match=bool(journal) and search_calls==m['counts']['full_surrogate_search_calls'] and validations==m['counts']['selected_validation_calls'],target_event_gates=gates,
        all_reference_ids_match=all(timeline.at(float(r['reference_elapsed_s']))[1]['id']==r['window_id'] for r in rows),
        C2_handoffs_reconstructed=True,source_snapshot_intact=source_ok,current_decoder_matches_snapshot=decoder_ok,state_gap_s=float(np.diff(t).max()),
        input_sha256={str(folder/name):hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files if (folder/name).exists()})
    result['audit_passed']=bool(result['rmse_difference']<1e-10 and result['peak_difference']<1e-10 and result['worker_counts_match'] and all(g['passed'] for g in gates) and result['all_reference_ids_match'] and source_ok and decoder_ok)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if any(folder.resolve()==output.parent or folder.resolve() in output.parents for folder in args.folders):raise ValueError('审计输出须位于原始日志之外')
    rows=[audit_run(folder) for folder in args.folders];output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scope='Read-only recomputation; audit pass is evidence consistency, not mission/algorithm superiority',runs=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=len(rows),audit_passed=all(row['audit_passed'] for row in rows),output=str(output)),ensure_ascii=False))

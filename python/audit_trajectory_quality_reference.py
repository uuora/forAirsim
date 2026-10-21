"""Audit RGB/surface point semantics from existing flights; no AirSim access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_trajectory_matched_sweep import audit_sweep,descriptive
from trajectory_rgb_quality_reference import rgb_quality_reference,replay_centering_ranking


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(sweep,output):
    if output.exists():raise FileExistsError(f'Refuse to overwrite: {output}')
    summary,_=audit_sweep(sweep);rows=[]
    for run in summary['runs']:
        if run['ledger_status']!='TERMINAL_REPORT':continue
        path=Path(run['report_path']);raw=json.loads(path.read_text(encoding='utf-8-sig'))
        frames={f['frame_id']:f for f in raw['observations']};plans=[]
        for plan in raw['planning_records']:
            frame=frames[plan['frame_id']];meta=frame['timestamped_observation']
            reference=rgb_quality_reference(frame['target_estimate_world_ned_m'],frame['actual_camera_world_ned_m'],
                meta['centre_px'],meta['projection']['rgb_size_wh'],meta['projection']['horizontal_fov_deg'])
            replay=replay_centering_ranking(plan['plan'],reference['offset_from_surface_ned_m'])
            difference=float(np.linalg.norm(np.array(reference['centre_px'])-np.array(reference['surface_anchor_px'])))
            plans.append(dict(plan_id=plan['plan_id'],frame_id=frame['frame_id'],pixel_reference_gap_norm_px=difference,
                reference=reference,offline_replay=replay))
        gap=descriptive(row['pixel_reference_gap_norm_px'] for row in plans)
        gap['maximum']=max((row['pixel_reference_gap_norm_px'] for row in plans),default=None)
        rows.append(dict(policy=run['policy'],condition_id=run['condition_id'],repetition=run['repetition'],
            report_path=str(path.resolve()),report_sha256=digest(path),task_success=run['task_success'],
            planning_records=len(plans),pixel_reference_gap_norm_px=gap,
            offline_selected_index_change_count=sum(row['offline_replay']['original_selected_index']!=row['offline_replay']['proposed_selected_index'] for row in plans),
            original_no_feasible_count=sum(row['offline_replay']['feasible_candidate_count']==0 for row in plans),plans=plans))
    result=dict(kind='RGB_CENTROID_VS_DEPTH_SURFACE_REFERENCE_OFFLINE_AUDIT',counts_as_flight_data=False,
        sweep_path=str(sweep.resolve()),sweep_sha256=digest(sweep),runs=rows,
        source=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),
        dependency_sha256=digest(Path(__file__).with_name('trajectory_rgb_quality_reference.py')),
        limits=['Offline ranking changes do not prove actual flight outcome or causal responsibility for historical failures',
            'The RGB reference lies on a representative axial-depth plane; it is not the true physical balloon center',
            'Original feasibility, constraints, weights, failures and raw files are preserved'],
        ai_assistance='Read-only semantic audit assisted by AI')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([dict(policy=r['policy'],plans=r['planning_records'],gap=r['pixel_reference_gap_norm_px'],ranking_changes=r['offline_selected_index_change_count']) for r in rows]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.sweep,args.output)

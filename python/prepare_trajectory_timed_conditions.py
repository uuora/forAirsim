"""Freeze V6 diagnostic gates and two-order, three-condition development pilots."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FLIGHT_SOURCES=('single_drone_experiment.py','visual_servo.py','target_provider.py','balloon_detector.py',
    'trajectory_single_experiment.py','perception_trajectory_planner.py','timestamped_target_provider.py',
    'matched_trajectory_start.py','trajectory_execution_guard.py','trajectory_timed_reference.py','trajectory_timed_experiment.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Refuse to overwrite: {output}')
    parent=ROOT/'configs/trajectory_guarded_pilot_v5/conditions.json'
    old=json.loads(parent.read_text(encoding='utf-8'))
    configs={}
    for entry in old['conditions']:
        source=Path(entry['configuration_path'])
        if sha(source)!=entry['configuration_sha256']:raise ValueError('Parent V5 freeze changed')
        cfg=json.loads(source.read_text(encoding='utf-8'))
        cfg.update(protocol_id='single_trajectory_time_indexed_v6_20261003',
            research_stage='fixed_height_time_indexed_reference_and_freshness_scheduling_development',
            footprint_anchor_contract='fixed_forward_projection_of_estimated_surface_point_not_detection_centroid')
        cfg['planning'].update(execution_reference_mode='time_indexed',
            timed_execution=dict(next_cycle_budget_s=.17,rpc_overhead_budget_s=.035,state_query_budget_s=.015))
        first=['efficiency','perception'] if entry['condition_id']=='target_y_plus_0p4' else ['perception','efficiency']
        cfg['condition_metadata'].update(planned_repetitions=[1,2],
            method_orders_by_repetition={'1':first,'2':list(reversed(first))},
            order_seed_scope='explicit_counterbalanced_engineering_order_not_scene_seed')
        cfg['evidence_limits'] += ['V6 elapsed host-time reference at every request; fixed world Z, not complete 3D',
            'Foreground yields preserve declared time for next capture/plan; timing budgets are estimates',
            'Local media writer drained before approach returns; all approach XY requests version/age traced',
            'Two paired orders per condition are development data, not formal method superiority evidence']
        configs[entry['condition_id']]=cfg
    output.mkdir()
    conditions=[]
    for condition,cfg in configs.items():
        path=output/(condition+'.json')
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        conditions.append(dict(condition_id=condition,configuration_path=str(path),configuration_sha256=sha(path),planned_repetitions=2))
    plans=[]
    for repetition in (1,2):
        for condition in ('center','target_y_plus_0p4','target_y_minus_0p4'):
            plans.append(dict(planned_session_index=len(plans)+1,condition_id=condition,repetition=repetition,
                configuration_path=str(output/(condition+'.json')),target_source='sensor_surface_estimate',
                method_order=configs[condition]['condition_metadata']['method_orders_by_repetition'][str(repetition)]))
    short=deepcopy(configs['center']);short['planning']['max_interface_planning_steps']=3
    short['condition_id']='center_short_time_indexed_gate'
    short_path=output/'center_short.json'
    short_path.write_text(json.dumps(short,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(schema_version=1,kind='V6_TIME_INDEXED_FRESHNESS_CONFIG_FREEZE',preparation_status='PREPARED_NOT_EXECUTED',
        formal_thesis_effect_results=False,prepared_local=datetime.now().astimezone().isoformat(),
        planned_sessions=len(plans),planned_flights=2*len(plans),conditions=conditions,session_plan=plans,
        separate_short_gate=dict(path=str(short_path),sha256=sha(short_path),planned_flights=2),
        parent_freeze=dict(path=str(parent),sha256=sha(parent)),
        current_flight_source_sha256={name:sha(ROOT/'python'/name) for name in FLIGHT_SOURCES},
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        gate='Short trace/motion gate, separate positive diagnostic gate, then frozen full development sweep',
        ai_assistance='Configuration preparation assisted by AI; no unrun outcome recorded')
    (output/'conditions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    positive=deepcopy(result)
    positive.update(kind='V6_SEPARATE_POSITIVE_DIAGNOSTIC_CONFIG_FREEZE',planned_sessions=1,planned_flights=2)
    positive['conditions']=[dict(next(c for c in conditions if c['condition_id']=='target_y_plus_0p4'),planned_repetitions=1)]
    positive['session_plan']=[dict(plans[1],planned_session_index=1)]
    (output/'positive_gate_conditions.json').write_text(json.dumps(positive,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(output/'conditions.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)

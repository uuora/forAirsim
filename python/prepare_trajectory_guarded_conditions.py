"""Freeze V5 envelope pilots separately from unchanged V4 inputs/results."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from prepare_trajectory_matched_conditions import validate_base

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f"Refuse to overwrite: {output}")
    parent=ROOT/'configs/trajectory_matched_pilot_v4/conditions.json'
    old=json.loads(parent.read_text('utf-8'))
    sources={c['condition_id']:c for c in old['conditions']}
    configs={}
    for condition,source in sources.items():
        path=Path(source['configuration_path'])
        if sha(path)!=source['configuration_sha256']:raise ValueError('V4 frozen source changed')
        cfg=json.loads(path.read_text('utf-8'))
        validate_base(json.loads(Path(sources['center']['configuration_path']).read_text('utf-8')))
        cfg['protocol_id']='single_trajectory_guarded_v5_20261003'
        cfg['research_stage']='fixed_height_measured_footprint_and_execution_envelope_development'
        cfg['condition_metadata']['planned_repetitions']=[1]
        cfg['condition_metadata']['method_orders_by_repetition']={'1':['perception','efficiency'] if condition!='target_y_plus_0p4' else ['efficiency','perception']}
        cfg['condition_metadata']['order_seed_scope']='explicit_staged_gate_order_not_scene_seed'
        cfg['planning'].update(command_duration_s=.15,execution_sample_dt_s=.05,
            execution_envelope=dict(horizon_s=.5,position_reserve_m=.02,freshness_reserve_s=.06,
                minimum_command_s=.03,brake_timeout_s=4.,max_freshness_replans=6,
                image_border_margin_fraction=.08))
        cfg['evidence_limits']=[e for e in cfg['evidence_limits'] if 'Two repetitions' not in e and 'Order seed' not in e]+[
            'One pair per condition; separate short gate; not formal method effectiveness data',
            'Observed bbox rectangle at representative axial depth, not true 3D target extent',
            'Execution envelope assumes declared delay and braking acceleration; no physical safety guarantee',
            'Shortened requests and adaptive leases; original .15 m/s, .30 m/s2 and .50s age budgets retained']
        configs[condition]=cfg
    output.mkdir(parents=True,exist_ok=False)
    conditions=[];plan=[]
    for index,condition in enumerate(('center','target_y_plus_0p4','target_y_minus_0p4'),1):
        path=output/(condition+'.json')
        path.write_text(json.dumps(configs[condition],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        conditions.append(dict(condition_id=condition,configuration_path=str(path),configuration_sha256=sha(path),planned_repetitions=1))
        plan.append(dict(planned_session_index=index,condition_id=condition,repetition=1,
            configuration_path=str(path),target_source='sensor_surface_estimate',
            method_order=configs[condition]['condition_metadata']['method_orders_by_repetition']['1']))
    short=deepcopy(configs['center']);short['planning']['max_interface_planning_steps']=3
    short['condition_id']='center_short_guarded_gate';short['protocol_id']+='_separate_short_gate'
    short_path=output/'center_short.json'
    short_path.write_text(json.dumps(short,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt=dict(schema_version=1,kind='V5_EXECUTION_ENVELOPE_CONFIG_FREEZE',preparation_status='PREPARED_NOT_EXECUTED',
        formal_thesis_effect_results=False,prepared_local=datetime.now().astimezone().isoformat(),
        planned_sessions=3,planned_flights=6,conditions=conditions,session_plan=plan,
        separate_short_gate=dict(path=str(short_path),sha256=sha(short_path),planned_flights=2),
        parent_v4_freeze=dict(path=str(parent),sha256=sha(parent)),
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        gate='Short matched-start and sampled physical motion gate, then staged center/positive/negative pilots',
        ai_assistance='Engineering configuration preparation assisted by AI; no unrun result recorded')
    path=output/'conditions.json'
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/trajectory_guarded_pilot_v5')
    args=parser.parse_args();prepare(args.output)

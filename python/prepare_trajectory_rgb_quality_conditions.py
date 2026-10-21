"""Freeze V6.2 RGB-centroid quality semantics with common unchanged budgets."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from prepare_trajectory_timed_conditions import FLIGHT_SOURCES,ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Refuse to overwrite: {output}')
    parent=ROOT/'configs/trajectory_timed_pilot_v6p1/conditions.json'
    old=json.loads(parent.read_text(encoding='utf-8-sig'));configs={}
    sources={name:sha(ROOT/'python'/name) for name in (*FLIGHT_SOURCES,'trajectory_rgb_quality_reference.py')}
    for entry in old['conditions']:
        source=Path(entry['configuration_path'])
        if sha(source)!=entry['configuration_sha256']:raise ValueError('Parent V6.1 freeze changed')
        cfg=json.loads(source.read_text(encoding='utf-8-sig'))
        cfg.update(protocol_id='single_trajectory_rgb_centroid_quality_v6p2_20261003',
            research_stage='fixed_height_time_indexed_rgb_centroid_quality_development')
        cfg['planning']['image_center_reference_mode']='rgb_detection_centroid_axial_plane'
        cfg['flight_source_sha256']=sources
        cfg['evidence_limits'] += ['RGB centering reference lies on representative surface axial-depth plane; not true physical balloon center',
            'Surface distance, target footprint, candidates, physical constraints and weights preserve the parent contract']
        configs[entry['condition_id']]=cfg
    output.mkdir();conditions=[]
    for condition,cfg in configs.items():
        path=output/(condition+'.json')
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        conditions.append(dict(condition_id=condition,configuration_path=str(path),configuration_sha256=sha(path),planned_repetitions=2))
    plans=[dict(row,configuration_path=str(output/(row['condition_id']+'.json'))) for row in old['session_plan']]
    short=deepcopy(configs['center']);short['condition_id']='center_short_rgb_centroid_quality_gate'
    short['planning']['max_interface_planning_steps']=3;short_path=output/'center_short.json'
    short_path.write_text(json.dumps(short,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(schema_version=1,kind='V6P2_RGB_CENTROID_QUALITY_REFERENCE_CONFIG_FREEZE',preparation_status='PREPARED_NOT_EXECUTED',
        formal_thesis_effect_results=False,prepared_local=datetime.now().astimezone().isoformat(),
        planned_sessions=6,planned_flights=12,conditions=conditions,session_plan=plans,
        separate_short_gate=dict(path=str(short_path),sha256=sha(short_path),planned_flights=2),
        parent_freeze=dict(path=str(parent),sha256=sha(parent)),
        current_flight_source_sha256=sources,
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        physical_budget_changes='NONE; same candidates, constraints, weights, task stopping criterion, motion and age limits',
        semantic_change='Projection centering evaluates measured RGB centroid bearing on the same axial-depth plane; surface distance/footprint retain original anchors',
        motion_gate='Post-capture actual state checked before planning; recorded inner-margin violations independently stop following sessions',
        gate='Pure and shared-input regression, independent short gate, separate positive gate, frozen full development sweep',
        ai_assistance='Preparation assisted by AI; no unexecuted outcome recorded')
    (output/'conditions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    positive=deepcopy(result)
    positive.update(kind='V6P2_SEPARATE_POSITIVE_DIAGNOSTIC_CONFIG_FREEZE',planned_sessions=1,planned_flights=2)
    positive['conditions']=[dict(next(c for c in conditions if c['condition_id']=='target_y_plus_0p4'),planned_repetitions=1)]
    positive['session_plan']=[dict(plans[1],planned_session_index=1)]
    (output/'positive_gate_conditions.json').write_text(json.dumps(positive,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(output/'conditions.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)

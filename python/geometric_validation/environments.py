"""Reproducible control-task presets; loading and selection never use AirSim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'configs/geometric_environments_v1.json'


def load_catalog(path=CATALOG):
    catalog=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if catalog.get('schema')!='geometric_environments_v1':
        raise ValueError('Unknown environment catalog')
    ids=[e['id'] for e in catalog['environments']]
    if len(set(ids))!=len(ids) or catalog['default_environment'] not in ids:
        raise ValueError('Duplicate environment or missing default')
    for entry in catalog['environments']:
        config_path=(ROOT/entry['config']).resolve()
        if not config_path.is_relative_to(ROOT/'configs'):
            raise ValueError('Environment config must stay in project configs')
        if 'task' in entry:
            task=entry['task'];points=np.asarray(task['points_ned'],dtype=float)
            durations=np.asarray(task['initial_segment_durations'],dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<3 or not np.isfinite(points).all():
                raise ValueError('Environment needs finite 3D waypoints with an intermediate pass')
            if durations.shape!=(len(points)-1,) or not np.isfinite(durations).all() or np.any(durations<=0):
                raise ValueError('Invalid environment initial durations')
            if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1.):
                raise ValueError('Environment waypoints must be separated by at least one metre')
            if np.any(np.linalg.norm(points[2:]-points[:-2],axis=1)<1e-6):
                raise ValueError('Exact reversal has an undefined shared tangent')
    return catalog


def get_environment(environment_id, catalog_path=CATALOG):
    for entry in load_catalog(catalog_path)['environments']:
        if entry['id']==environment_id:
            return entry
    raise ValueError('Unknown environment: '+str(environment_id))


def config_for(environment_id):
    return (ROOT/get_environment(environment_id)['config']).resolve()


def load_environment_config(environment_id, catalog_path=CATALOG):
    catalog=load_catalog(catalog_path);entry=get_environment(environment_id,catalog_path)
    cfg=json.loads((ROOT/entry['config']).read_text(encoding='utf-8-sig'))
    if 'task' in entry:
        common=cfg.get('comparison',{});metadata=common.get('environment',{})
        base=json.loads((ROOT/catalog['base_task']).read_text(encoding='utf-8'))
        control=json.loads((ROOT/base['base_config']).read_text(encoding='utf-8'))
        valid=(common.get('task')==entry['task'] and metadata.get('id')==environment_id
               and metadata.get('catalog_sha256')==hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
               and all(common.get(k)==base[k] for k in ('time_search','passage'))
               and all(cfg.get(k)==control[k] for k in ('controller','limits','execution','airsim','visualization')))
        if not valid:
            raise ValueError('Frozen environment plan is stale; run --prepare-environments before flight')
    return cfg


def task_summary(config):
    task=config.get('comparison',{}).get('task')
    if task is None:
        return None
    points=np.asarray(task['points_ned'],dtype=float)
    steps=np.diff(points,axis=0)
    angles=[]
    for a,b in zip(steps[:-1],steps[1:]):
        angles.append(float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1.,1.)))))
    return dict(polyline_length_m=float(np.linalg.norm(steps,axis=1).sum()),
                altitude_min_m=float(-points[:,2].max()),altitude_max_m=float(-points[:,2].min()),
                intermediate_points=len(points)-2,max_direction_change_deg=max(angles,default=0.))


def prepare_environment(environment_id, catalog_path=CATALOG):
    from .campaign import prepare_spec
    from .runtime import load_config

    catalog=load_catalog(catalog_path);entry=get_environment(environment_id,catalog_path)
    target=(ROOT/entry['config']).resolve()
    if 'task' not in entry:
        load_config(target)
        return target
    spec=deepcopy(json.loads((ROOT/catalog['base_task']).read_text(encoding='utf-8')))
    spec['task']=deepcopy(entry['task'])
    spec['environment']=dict(id=entry['id'],label=entry['label'],focus=entry['focus'],
        catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    target.parent.mkdir(parents=True,exist_ok=True)
    return prepare_spec(spec,target,fingerprint)


def prepare_all(catalog_path=CATALOG):
    return [prepare_environment(e['id'],catalog_path) for e in load_catalog(catalog_path)['environments']]


def matching_comparison(config, reports_root=None):
    from dual_target_control.machine_config import apply_machine_config
    from .planners import common_fingerprint

    if 'comparison' not in config:
        return None
    identity=common_fingerprint(apply_machine_config(config))
    pages=sorted(Path(reports_root or ROOT/'reports').glob('geometric_three_planners_*/index.html'),reverse=True)
    for page in pages:
        try:
            old=json.loads((page.parent/'config_frozen.json').read_text(encoding='utf-8'))
            if 'comparison' in old and common_fingerprint(apply_machine_config(old))==identity:
                return page
        except (ValueError,KeyError,OSError):
            continue
    return None

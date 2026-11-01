"""Explicit research variants; production six-method menu retains defaults."""
from copy import deepcopy
from pathlib import Path
import json,numpy as np
from .online_curve import fallback,reference_checks
from .online_live import OnlineRuntime

ARMS={
 'O':dict(label='普通Hermite',method='hermite_ordinary',progress_guard=False,shape_scale=1.),
 'A':dict(label='基础ABOM',method='hermite_abom',progress_guard=False,shape_scale=1.),
 'P':dict(label='原配对候选',method='hermite_abom_improved_candidate',progress_guard=False,shape_scale=1.),
 'AG':dict(label='基础ABOM+到达时间约束',method='hermite_abom',progress_guard=True,shape_scale=1.),
 'PG':dict(label='配对候选+到达时间约束',method='hermite_abom_improved_candidate',progress_guard=True,shape_scale=1.),
 'PGL':dict(label='配对候选+时间约束+局部形状',method='hermite_abom_improved_candidate',progress_guard=True,shape_scale=.25),
 'AGL':dict(label='基础ABOM+时间约束+局部形状',method='hermite_abom',progress_guard=True,shape_scale=.25),
 'RGL':dict(label='高斯搜索+时间约束+局部形状',method='hermite_abom',progress_guard=True,shape_scale=.25,proposal_mode='gaussian'),
}


def prepare_request(request,active,handoff,settings):
    request=deepcopy(request);scale=float(settings['shape_scale'])
    if not 0<scale<=1:raise ValueError('形状域倍率须在(0,1]')
    vector,check=fallback(request);request['anchor']=vector
    for i in (0,1,2,4,5,6):request['lower'][i]*=scale;request['upper'][i]*=scale
    request['search_settings']={k:settings[k] for k in ('paired_scale','paired_fraction','dropout','learning_rate','proposal_mode') if k in settings}
    if settings['progress_guard']:
        ceiling=active['start_s']+float(active['_curve'].ends[0]);remaining=ceiling-handoff
        if remaining<=3.+1e-5:return None,'REMAINING_TIME_BELOW_PARAMETER_DOMAIN'
        request['upper'][3]=min(7.,remaining);request['anchor'][3]=min(request['anchor'][3],request['upper'][3])
        request['goal_arrival_ceiling_s']=ceiling
        if not reference_checks(request,request['anchor'])['passed']:return None,'NO_KINEMATIC_ANCHOR_WITHIN_DEADLINE'
    return request,None


def config_for(arm,seed=1301,duration=25.,budget=.8,max_candidates=100000):
    from online_curve_panel import online_config
    if arm not in ARMS:raise ValueError('未知消融组')
    setting=deepcopy(ARMS[arm]);setting.update(arm=arm,paired_scale=.15,paired_fraction=.5,dropout=.95,learning_rate=.001)
    cfg=online_config(seed,duration,setting['method']);cfg['online'].update(research=setting,study_id='progress_locality_v1',budget_s=budget,max_candidates=max_candidates)
    return cfg


class ResearchRuntime(OnlineRuntime):
    research_source_paths=[Path(__file__),Path(__file__).with_name('online_results.py')]

    def _request(self,boundary):
        request=super()._request(boundary)
        request['seed']=int(self.config['online'].get('optimizer_seed',self.config['online']['seed']))+self.route_index
        return request

    def _begin_run(self):
        super()._begin_run()
        path=self._run_dir/'online_manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
        manifest['research_treatment']=self.config['online']['research']
        # Preserve full fingerprint; this extra digest removes only declared
        # algorithm-treatment settings for explicitly paired study comparisons.
        import hashlib
        fixed=deepcopy(manifest['conditions']);fixed['online'].pop('research',None)
        manifest['study_control_fingerprint']=hashlib.sha256(json.dumps(dict(conditions=fixed,sources=manifest['source_sha256']),sort_keys=True).encode()).hexdigest()
        manifest['study_removed_fields']=['conditions.online.research (algorithm treatment)','method (already excluded in full condition fingerprint)']
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    def _prepare_search_request(self,request,active,handoff):
        prepared,reason=prepare_request(request,active,handoff,self.config['online']['research'])
        if reason:self._event('SEARCH_SKIPPED_PROGRESS',reason=reason,goal_arrival_s=active['start_s']+float(active['_curve'].ends[0]),handoff_s=handoff)
        return prepared

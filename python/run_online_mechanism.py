"""Run one explicitly selected research arm, not the whole campaign."""
import argparse,json
from pathlib import Path
from dual_target_control.online_study import ARMS,config_for
from run_control_full_flow import trial,save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--airsim',action='store_true',required=True)
    p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--scene-seed',type=int,default=1401)
    p.add_argument('--optimizer-seed',type=int,default=21);p.add_argument('--duration',type=float,default=40.)
    p.add_argument('--budget',type=float,default=.8);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=config_for(a.arm,a.scene_seed,a.duration,a.budget);cfg['online'].update(optimizer_seed=a.optimizer_seed,study_id='progress_locality_repeated_v2')
    task=dict(id=f'online_{a.scene_seed}_{a.optimizer_seed}_{a.arm}',family='online',arm=a.arm,method=cfg['online']['method'],scene_seed=a.scene_seed,optimizer_seed=a.optimizer_seed,config=cfg)
    save(a.output/'request_frozen.json',task);result=trial(task,a.output/'run');save(a.output/'result.json',result)
    if result.get('run_dir'):
        from audit_online_control import audit_run
        try:save(a.output/'audit.json',audit_run(result['run_dir']))
        except Exception as ex:save(a.output/'audit.json',dict(audit_passed=False,error=repr(ex)))
    print(json.dumps(dict(status=result['status'],run_dir=result.get('run_dir'),closed_safely=result['closed_safely']),ensure_ascii=False))

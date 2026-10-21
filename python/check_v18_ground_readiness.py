"""Live ground-only observer/cleanup smoke test. Never issues flight commands."""
import argparse
import time,os
from pathlib import Path
from types import SimpleNamespace
from fixed_map_cpu_runtime import configure_from_argv
configure_from_argv()
import numpy as np
from fixed_map_perception import ROOT,read,save,utc,sha
from fixed_map_scene_binding import bind_scene_config
from fixed_map_perception_process import PerceptionProcess
from fixed_map_trial_cleanup import probe_ground,finalize_trial


def run(output, config='configs/fixed_map_perception_yolo_cpu_v18.json'):
    output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    before=probe_ground()
    if not before['passed']:raise ValueError('GROUND_NOT_RELEASED')
    config_path=ROOT/config
    cfg=bind_scene_config(read(config_path),read(ROOT/'configs/fixed_map_scene_state.json'))
    assert cfg['ground_development_only'] and not cfg['flight_authorized']
    save(output/'DECLARATION.json',dict(utc=utc(),scope='live ground observer and shutdown only; zero aircraft-motion commands',config=cfg,config_sha256=sha(config_path),ground=before))
    service=PerceptionProcess(output/'online',cfg);issue=None;resource_trace=None
    if cfg.get('resource_trace_v20'):
        from fixed_map_resources import ResourceTrace
        resource_trace=ResourceTrace(output/'resource_trace.jsonl',lambda:dict(stage='GROUND_OBSERVER_ONLY'),lambda:dict(driver=os.getpid(),sensor=service.process.pid if hasattr(service,'process') else None,engine=cfg['runtime_engine_identity']['pid']))
        resource_trace.start()
    try:
        service.start();deadline=time.monotonic()+40
        while not service.warm_ready.is_set() and time.monotonic()<deadline:
            if service.process.poll() is not None:raise ValueError('GROUND_SENSOR_WORKER_EXITED')
            time.sleep(.1)
        if not service.warm_ready.is_set():raise ValueError('GROUND_WARMUP_DEADLINE')
        time.sleep(12)
    except Exception as ex:issue=str(ex)
    executor=SimpleNamespace(state={},is_alive=lambda:False)
    coordinator=SimpleNamespace(phase='GROUND_OBSERVER_ONLY',status=lambda:dict(outcome='GROUND_OBSERVER_ONLY'))
    terminal=finalize_trial(output,executor,coordinator,service,issue,False)
    ages=[r['source_age_on_parent_receipt_s'] for r in service.publication_metrics]
    result=dict(utc=utc(),ground_verified=terminal['ground_verification_passed'],cleanup_issue=terminal['issue'],worker_exited=not hasattr(service,'process') or service.process.poll() is not None,worker_started=hasattr(service,'process'),publications=len(ages),max_parent_age_s=max(ages,default=None),median_parent_age_s=float(np.median(ages)) if ages else None,fresh_publications=sum(a<=cfg['current_max_age_s'] for a in ages),current_max_age_s=cfg['current_max_age_s'],aircraft_motion_commands=0,weights_unchanged=True,scope='ground observer throughput/cleanup only, not live mission acceptance')
    if resource_trace is not None:
        result['resource_trace']=resource_trace.stop()
        if result['resource_trace']['status']!='PASS':result['cleanup_issue']=result['cleanup_issue'] or 'RESOURCE_TRACE_FAILED'
    save(output/'GROUND_CHECK.json',result);print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--config',default='configs/fixed_map_perception_yolo_cpu_v18.json');args=parser.parse_args();run(args.output,args.config)

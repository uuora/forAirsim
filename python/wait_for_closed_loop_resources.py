"""Bounded read-only resource wait; never starts a model or aircraft."""
import argparse
import json
import time
from pathlib import Path
from fixed_map_resources import memory_headroom


def wait(output,seconds):
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    deadline=time.monotonic()+seconds;stable=0;rows=[]
    while time.monotonic()<deadline:
        memory=memory_headroom();good=min(memory['physical_available_bytes'],memory['commit_available_bytes'])>=3*2**30
        stable=stable+1 if good else 0
        row=dict(host_time_ns=time.time_ns(),monotonic_s=time.monotonic(),memory=memory,original_model_load_threshold_bytes=3*2**30,ready=good,consecutive_ready_samples=stable)
        rows.append(row)
        with (output/'resource_samples.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(row)+'\n')
        if stable>=3:break
        time.sleep(1)
    result=dict(status='READY' if stable>=3 else 'INSUFFICIENT_RESOURCE_HEADROOM',last=rows[-1],samples=len(rows),aircraft_commands=0,model_loads=0,scope='Only resource readiness; fresh ground, scene and model checks still required')
    (output/'WAIT_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--seconds',type=int,default=60);args=parser.parse_args()
    if not 3<=args.seconds<=120:raise ValueError('RESOURCE_WAIT_OUTSIDE_BOUNDS')
    wait(args.output,args.seconds)

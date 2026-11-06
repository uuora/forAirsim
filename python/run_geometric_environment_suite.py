"""Explicit E0-E4 flights or read-only collection; default is offline checks."""
import argparse
import json
from pathlib import Path
import signal
import threading

from geometric_validation.environment_suite import run_suite, collect_suite
from geometric_validation.environments import load_catalog, config_for, load_environment_config
from geometric_validation.runtime import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--airsim',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--collect',type=Path)
    args=p.parse_args()
    if args.launch and not args.airsim:p.error('--launch requires --airsim')
    if args.collect:
        result=collect_suite(args.collect)
    elif args.airsim:
        stop=threading.Event()
        def request_stop(*_):
            stop.set();print('Stop requested; waiting for verified cleanup.',flush=True)
        signal.signal(signal.SIGINT,request_stop)
        folder=run_suite(args.output,args.launch,stop)
        result=collect_suite(folder);print(str(folder),flush=True)
    else:
        for entry in load_catalog()['environments']:
            load_environment_config(entry['id']);load_config(config_for(entry['id']))
        print(json.dumps(dict(check='PASS',environments=5,planned_flights=15,rpc=False)));return 0
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','environments')},ensure_ascii=True,indent=2))
    return 0 if result['accepted']==result['requested'] and result['reports_match_reaudit'] else 1


if __name__=='__main__':raise SystemExit(main())

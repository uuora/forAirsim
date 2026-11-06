"""Explicit standalone control commissioning. Default is a non-RPC check."""
import argparse
import json
from pathlib import Path
from geometric_validation.runtime import DEFAULT_CONFIG, load_config, local_preflight, run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--case',choices=['hover','line','turn','waypoints'],default='hover')
    p.add_argument('--check',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--airsim',action='store_true')
    p.add_argument('--launch',action='store_true');p.add_argument('--suite',action='store_true');p.add_argument('--audit',type=Path);p.add_argument('--output-root')
    a=p.parse_args()
    if a.launch and not a.airsim:p.error('--launch requires --airsim; --preflight only inspects an already running simulator')
    if a.audit:
        from geometric_validation.audit import audit
        result=audit(a.audit);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['data_audit_passed'] else 1
    cfg=load_config(a.config)
    if not a.airsim:
        result=local_preflight(cfg)
        if a.preflight:
            # Read-only probe; never takes ownership, arms, or launches a simulator.
            import tempfile
            from geometric_validation.runtime import connect,inspect_live
            with tempfile.TemporaryDirectory(prefix='geometric_preflight_') as tmp:
                c,_=connect(cfg,Path(tmp),launch=False);result['live']=inspect_live(c,cfg,Path(tmp))
        print(json.dumps(dict(check='PASS',rpc=a.preflight,details=result),ensure_ascii=False,indent=2));return 0
    cases=['hover','line','turn','waypoints'] if a.suite else [a.case]
    results=[]
    for case in cases:
        result=run(a.config,case,a.output_root,launch=a.launch);results.append(result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if not result['report']['accepted']:break  # escalate only after prior stage passes
    if a.suite:
        from datetime import datetime
        out=Path(a.output_root or Path(__file__).resolve().parents[1]/'reports');out.mkdir(parents=True,exist_ok=True)
        file=out/('geometric_suite_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        file.write_text(json.dumps(dict(requested_cases=cases,results=results,not_run=cases[len(results):]),ensure_ascii=False,indent=2),encoding='utf-8')
        print(str(file))
    return 0 if all(r['report']['accepted'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())

"""Compose version-specific immutable audits, keeping all earlier failures."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import analyze_t1_settlement as v7
import analyze_t1_settlement_v1p1 as v7p1
import analyze_t1_settlement_v1p2 as v7p2
import analyze_t1_settlement_v1p4 as v7p3


def compose(roots, output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('NEW_COMPOSED_ANALYSIS_REQUIRED')
    runs=[]
    for root in roots:
        arch=v7.read_json(Path(root)/'flight/report.json')['control_architecture']
        route=(v7p3 if arch.endswith('V7P3') else v7p2 if arch.endswith('V7P2')
            else v7p1 if arch.endswith('V7P1') else v7)
        runs.append(route.analyze_run(root))
    output.mkdir(parents=True)
    sources=('compose_t1_settlement_evidence.py','analyze_t1_settlement.py','analyze_t1_settlement_v1p1.py',
        'analyze_t1_settlement_v1p2.py','analyze_t1_settlement_v1p3.py','analyze_t1_settlement_v1p4.py',
        'audit_t1_feedback_stop_v1p1.py','t1_stop_receipt_match.py','t1_settlement_plan_v1.py',
        't1_settlement_plan_v1p1.py','t1_settlement_plan_v1p2.py','t1_settlement_plan_v1p3.py',
        't1_control_settlement.py','t1_identity_motion_policy_v1p1.py','t1_async_control_v1p4.py',
        'analyze_t1_delayed_identity.py','analyze_t1_identity_flight.py','analyze_t1_motion_scale.py','analyze_t1_async_timing.py')
    for name in sources:
        dest=output/'source_snapshots'/name; dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parent/name,dest)
    result=dict(kind='T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,new_flights_started_by_analysis=0,formal_efficacy_results=False,
        version_specific_auditors_preserved=True,
        limitations=['Each failed source version remains failed in its own audit.',
            '20-second scale failures are retained; 60-second runs use a separate predeclared duration.',
            'Engineering stops, same-source development repeats and split timing do not complete formal method comparisons.'])
    v7.save(output/'report.json',result)
    v7.save(output/'artifact_receipt.json',dict(files=[dict(path=str(p),sha256=v7.digest(p)) for p in output.rglob('*') if p.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:
        print(r['session_id'],r['base_evidence']['independent_integration_qualification'],r['feedback_stop_audit']['status'],r['endpoint_scale_status'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();compose(args.runs,args.output)

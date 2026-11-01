"""No simulator connection: re-audit restored current records at this path."""
import argparse,hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from dual_target_control.reproduction_paths import workspace_record_path
from dual_target_control.curve_audit import audit_run as audit_fixed
from audit_online_control import audit_run as audit_online

def verify(output):
    campaign=json.loads((ROOT/'reports/control_full_flow_20261101_v1/campaign.json').read_text(encoding='utf-8'));receipts=[];fixed_success=online_success=0
    for trial in campaign['trials']:
        folder=workspace_record_path(trial['run_dir']);online=trial['family']=='online';receipt=audit_online(folder) if online else audit_fixed(folder)
        if online:
            m=json.loads((folder/'online_metrics.json').read_text(encoding='utf-8'));success=m['outcome']=='DURATION_COMPLETE' and m['tracking_checks_passed'] and receipt['audit_passed'];online_success+=bool(success)
        else:success=receipt['outcome']=='COMPLETED' and receipt.get('control_passed') and receipt['status']=='PASS';fixed_success+=bool(success)
        receipts.append(dict(task=trial['task_id'],task_success=bool(success),**receipt))
    result=dict(status='PASS' if fixed_success==12 and online_success==16 and all(r.get('audit_passed',r.get('status')=='PASS') for r in receipts) else 'FAIL',
        checkout=str(ROOT),fixed_success=fixed_success,online_success=online_success,records=len(receipts),runs=receipts,
        scope='Restored historical evidence recomputation only; no new flight or independent-host effectiveness claim')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False))
    return result['status']=='PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports'/('reproduction_check_'+datetime.now().strftime('%Y%m%d_%H%M%S'))/'audit.json');a=p.parse_args();raise SystemExit(0 if verify(a.output) else 1)

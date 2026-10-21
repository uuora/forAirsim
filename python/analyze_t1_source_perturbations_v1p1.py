"""Preserve V8 runtime; repair the old audit's HTTP/pending-review conflation."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import analyze_t1_source_perturbations as original
from t1_pending_review_audit_v1 import pending_review_audit


def analyze_run(root):
    r=original.analyze_run(root)
    raw=original.read_json(Path(root)/'flight/report.json')
    audit=pending_review_audit(raw)
    old=r['review_handoff']['issues']
    passed={row['frame_id'] for row in audit['rows'] if row['status']=='PASS_SAVED_PENDING_REVIEW_CHAIN'}
    repaired=[x for x in old if x['issue']=='PENDING_CARRY_NOT_BOUND_TO_ACTUAL_CURRENT_REVIEW' and x['frame_id'] in passed]
    remaining=[x for x in old if x not in repaired]
    r['review_handoff']['issues']=remaining+audit['issues']
    r['review_handoff']['retained_pending_review_audit']=audit
    r['review_handoff']['legacy_audit_rows_replaced_by_explicit_saved_chain_proof']=repaired
    r['review_handoff']['original_audit_issues_preserved']=old
    if r['independent_fault_qualification']=='FAIL_SOURCE_FAULT_EVIDENCE' and repaired and not remaining and not audit['issues']:
        b=r['base_evidence']
        evidence=(b['independent_integration_qualification']=='PASS_CONTROLLED_DEVELOPMENT_SCOPE'
            and r['feedback_stop_audit']['status'].startswith('PASS') and r['zero_hold_audit']['status'].startswith('PASS')
            and r['actual_clear_admission_audit']['status'].startswith('PASS')
            and r['reply_delivery_audit']['status'].startswith('PASS')
            and r['measured_grid_check']['status'].startswith('PASS') and r['physical_scene_audit']['status'].startswith('PASS'))
        if evidence:r['independent_fault_qualification']='FAIL_SOURCE_FAULT_BEHAVIOR' if r['source_fault_behavior']['issues'] else 'PASS_BOUNDED_SOURCE_FAULT_DEVELOPMENT'
    if audit['issues']:r['independent_fault_qualification']='FAIL_SOURCE_FAULT_EVIDENCE'
    return r


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_SAVED_PENDING_REVIEW_ANALYSIS_REQUIRED')
    runs=[analyze_run(p) for p in roots];output.mkdir(parents=True)
    names=('analyze_t1_source_perturbations_v1p1.py','t1_pending_review_audit_v1.py','analyze_t1_source_perturbations.py',
        't1_reply_delivery_fault_v1.py','t1_perturbation_contract_v1.py','audit_t1_feedback_stop_v1p1.py',
        't1_stop_receipt_match.py','analyze_t1_delayed_identity.py','analyze_t1_identity_flight.py',
        'analyze_t1_settlement.py','analyze_t1_settlement_v1p1.py','analyze_t1_settlement_v1p4.py',
        'analyze_t1_review_handoff_v5p1.py','t1_delayed_semantic_bridge.py')
    for name in names:
        p=Path(__file__).with_name(name);dest=output/'source_snapshots'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    result=dict(kind='T1_SOURCE_FAULT_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=['Original V8 runtime and initial failed analysis unchanged.',
            'Retained accepted review confirmation and the next HTTP request have distinct IDs.',
            'Artificial reply holds and natural service times are recorded separately.',
            'Fault behavior and motion scale do not establish formal efficacy or hard realtime.'])
    original.timing.save(output/'report.json',result)
    original.timing.save(output/'artifact_receipt.json',dict(files=[dict(path=str(p),sha256=original.digest(p)) for p in output.rglob('*') if p.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:print(r['session_id'],r['independent_fault_qualification'],r['source_fault_behavior']['identity_counts'],r['endpoint_scale_status'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.runs,a.output)

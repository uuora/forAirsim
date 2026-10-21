"""Retain actual spawn injection evidence; RPC fixture only, no simulator/API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time

from t1_process_control_v1p1 import ProcessControlV1P1 as ProcessControlV1
from test_t1_process_control_v1p1 import config, proposal, abrupt_parent
from audit_t1_process_control_v1p1 import audit


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def inject(kind, controller):
    previous = sys.getswitchinterval()
    started = time.monotonic()
    deadline = started+.65
    try:
        if kind == 'DETECTION_CPU_SUBSTITUTE':
            sys.setswitchinterval(1.)
            while time.monotonic()<deadline:
                pass
        elif kind == 'MEDIA_QUEUE_ADMISSION_SUBSTITUTE':
            bounded = queue.Queue(maxsize=1)
            bounded.put_nowait('occupied')
            try: bounded.put('new media', timeout=.65)
            except queue.Full: pass
        elif kind == 'PARENT_JSON_SERIALIZATION':
            sys.setswitchinterval(1.)
            values = dict(rows=[list(range(1024)) for _ in range(256)])
            while time.monotonic()<deadline:
                json.dumps(values)
        elif kind == 'RESULT_READER_SUSPENDED':
            controller._receiver_stop.set()
            controller._receiver.join(.5)
            time.sleep(max(0., deadline-time.monotonic()))
        else:
            raise ValueError('DECLARED_OFFLINE_INJECTION_REQUIRED')
    finally:
        sys.setswitchinterval(previous)
    return dict(boundary=kind, started_monotonic_s=started, completed_monotonic_s=time.monotonic(),
        declared_duration_s=.65, scope='offline parent operation; original YOLO/VLX not invoked')


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    names=('t1_process_control_v1p1.py','t1_process_control_factories_v1.py','t1_process_control_fixture_v1.py',
           'audit_t1_process_control_v1p1.py','t1_process_trace_v1.py','run_t1_process_offline_v1p1.py','t1_async_control_v1p8.py','t1_capture_diagnostics_v1.py')
    source_hashes={name:sha(Path(__file__).with_name(name)) for name in names}
    save(output/'before_source_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_hashes,planned_conditions=5,simulator_calls=0,provider_requests=0))
    rows=[]
    for kind in ('DETECTION_CPU_SUBSTITUTE', 'MEDIA_QUEUE_ADMISSION_SUBSTITUTE',
                 'PARENT_JSON_SERIALIZATION', 'RESULT_READER_SUSPENDED'):
        folder=output/kind.lower()
        controller=ProcessControlV1(config(),folder,factory='t1_process_control_fixture_v1:make_fixture')
        decision=injection=None
        error=None
        try:
            controller.start();time.sleep(.36)
            ticket=controller.submit(*proposal(),[.009,0.])
            decision=controller.await_decision(ticket)
            if not decision.get('command') or decision['command']['purpose']!='CURRENT_TARGET_TRAJECTORY':
                raise RuntimeError('OFFLINE_NONZERO_ACTUAL_DISPATCH_REQUIRED')
            injection=inject(kind,controller)
            controller.wait_until_settled();controller.close()
        except Exception as exc:
            error=type(exc).__name__
            try:controller.close(timeout=6.)
            except Exception:pass
        checked=audit(folder);save(folder/'audit.json',checked)
        final=controller.scheduling_receipt()
        proof=controller.receipts().get('control-1',{})
        command=decision.get('command',{}) if decision else {}
        record=dict(condition=kind,status='PASS' if error is None and checked['status'].startswith('PASS') else 'FAIL',
            injection=injection,error_type=error,
            pid=final.get('pid'),parent_pid=final.get('parent_pid'),
            dispatch_monotonic_s=command.get('dispatch_monotonic_s'),
            actual_clear_monotonic_s=proof.get('target_setpoint_cleared_monotonic_s'),
            original_target_deadline_monotonic_s=command.get('target_end_deadline_monotonic_s'),
            completed_stop_monotonic_s=proof.get('completed_monotonic_s'),
            target_lease_renewed=proof.get('target_lease_renewed'),
            stop_verified=final.get('sampled_stop_verified'),transfer_permitted=controller.can_transfer_rpc(),
            final_receipt=dict(path=str(folder/'final.json'),sha256=sha(folder/'final.json')),
            audit=dict(path=str(folder/'audit.json'),sha256=sha(folder/'audit.json')))
        if record['actual_clear_monotonic_s'] is not None:
            record['clear_after_dispatch_s']=record['actual_clear_monotonic_s']-record['dispatch_monotonic_s']
            if record['actual_clear_monotonic_s']>record['original_target_deadline_monotonic_s']:
                record['status']='FAIL_ORIGINAL_TARGET_CLEAR_DEADLINE'
        save(folder/'injection.json',record);rows.append(record)
        print(kind,record['status'],flush=True)
        if controller.is_alive():raise RuntimeError('OFFLINE_OWNER_STILL_ACTIVE')
        controller.proposals.cancel_join_thread();controller.proposals.close();controller.responses.close()
    orphan=output/'parent_loss';orphan.mkdir()
    parent=mp.get_context('spawn').Process(target=abrupt_parent,args=(str(orphan),))
    parent.start();parent.join(15.)
    if parent.is_alive():raise RuntimeError('OFFLINE_PARENT_FIXTURE_STILL_ACTIVE')
    deadline=time.monotonic()+8.
    while not (orphan/'control/final.json').exists() and time.monotonic()<deadline:time.sleep(.02)
    checked=audit(orphan/'control');save(orphan/'audit.json',checked)
    final=json.loads((orphan/'control/final.json').read_text(encoding='utf-8'))
    rows.append(dict(condition='ACTUAL_PARENT_EXIT',status='PASS' if parent.exitcode==31 and
        final['close_reason']=='PARENT_PROCESS_EXITED' and checked['status'].startswith('PASS') else 'FAIL',
        parent_exit_code=parent.exitcode,stop_verified=final['sampled_stop_verified'],safe_landing_verified=False,
        final_receipt=dict(path=str(orphan/'control/final.json'),sha256=sha(orphan/'control/final.json')),
        audit=dict(path=str(orphan/'audit.json'),sha256=sha(orphan/'audit.json'))))
    result=dict(status='PASS_OFFLINE_SPAWN_INJECTIONS_V11P1' if all(r['status']=='PASS' for r in rows) else 'FAIL_OFFLINE_SPAWN_INJECTIONS',
        created_utc=datetime.now(timezone.utc).isoformat(),conditions=rows,new_flights=0,simulator_calls=0,
        source_sha256=source_hashes,
        provider_requests=0,real_YOLO_or_VLX_executed=False,hard_real_time_established=False,
        real_flight_latency_improvement_established=False,ai_assistance='implementation and evidence analysis')
    result['source_unchanged_after_execution']=all(sha(Path(__file__).with_name(n))==s for n,s in source_hashes.items())
    if not result['source_unchanged_after_execution']:result['status']='FAIL_OFFLINE_SOURCE_CHANGED'
    save(output/'report.json',result);print(result['status']);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raise SystemExit(0 if run(a.output)['status'].startswith('PASS') else 2)

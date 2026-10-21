"""Meaningful source-age, worker-delivery and frozen phase contract regressions."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace
import unittest

from t1_perturbation_contract_v1 import specification, validate_plan, ARCHITECTURE
from t1_reply_delivery_fault_v1 import ReplyDeliveryFaultBackend
from t1_scene_fault_v1 import glint_pixels
from t1_source_attribute_guard import band_evidence
from prepare_t1_identity_flight import phase_grid
from analyze_t1_source_perturbations import delivery_audit


class Backend:
    def __init__(self,clock):self.vlx_client=object();self.clock=clock;self.rows=[]
    def __call__(self,request):
        reply=SimpleNamespace(binding=request.binding(),completed_monotonic_s=self.clock[0],result={'real':True})
        self.rows.append({'request_binding':request.binding(),'backend_completed_monotonic_s':self.clock[0]})
        return reply
    def snapshot(self):return deepcopy(self.rows)


class SourcePerturbationTests(unittest.TestCase):
    def request(self,identity,receipt):
        return SimpleNamespace(frame=SimpleNamespace(readback_receipt_monotonic_s=receipt),
            binding=lambda:dict(request_id=identity,readback_receipt_monotonic_s=receipt))

    def wrapper(self):
        clock=[100.]
        def sleep(duration):clock[0]+=duration
        wrapper=ReplyDeliveryFaultBackend(Backend(clock),specification('delivery_delay','SHORT_XY')['reply_delivery_fault'],
            clock=lambda:clock[0],sleep=sleep)
        return wrapper,clock

    def test_real_reply_clock_is_preserved_during_hold(self):
        w,c=self.wrapper();w.activate(99.)
        request=self.request('first',100.);reply=w(request)
        self.assertEqual(reply.completed_monotonic_s,100.)
        self.assertEqual(reply.binding,request.binding());self.assertEqual(c[0],103.)
        record=w.snapshot()[0]['controlled_reply_delivery']
        self.assertEqual(record['declared_hold_s'],3.)
        self.assertEqual(record['delivered_to_policy_monotonic_s'],103.)
        self.assertFalse(record['natural_service_delay'])

    def test_acquisition_and_old_inflight_requests_do_not_consume_holds(self):
        w,c=self.wrapper();w(self.request('initial',98.));w.activate(100.)
        w(self.request('old-inflight',99.));w(self.request('new',100.))
        self.assertEqual([r['controlled_reply_delivery']['declared_hold_s'] for r in w.snapshot()],[0.,0.,3.])

    def test_only_two_frozen_holds_then_real_delivery(self):
        w,c=self.wrapper();w.activate(99.)
        for i in range(3):w(self.request(str(i),c[0]))
        self.assertEqual([r['controlled_reply_delivery']['declared_hold_s'] for r in w.snapshot()],[3.,9.,0.])
        self.assertEqual(c[0],112.)

    def test_epoch_cannot_be_rebased(self):
        w,c=self.wrapper();w.activate(99.)
        with self.assertRaises(ValueError):w.activate(100.)

    def plan(self,condition):
        base=Path(__file__).resolve().parents[1]/'configs/t1_settlement_visible60_v7p3_20261006/plan.json'
        p=json.loads(base.read_text(encoding='utf-8'));s=specification(condition,'SHORT_XY')
        p.update(control_architecture=ARCHITECTURE,source_perturbation=s,phases=s['phases'],planned_slots=phase_grid(s['phases'],.4))
        p['motion_scale_validation']['continuous_visible_duration_s']=None if condition=='occlusion' else 60.
        return p

    def test_actual_occlusion_grid_is_validated_separately(self):
        p=self.plan('occlusion');r=validate_plan(p)
        self.assertEqual(r['expected_main_slots'],150)
        self.assertEqual(sum(s['phase_id']=='hidden' for s in p['planned_slots']),10)
        p['planned_slots'][50]['phase_id']='visible'
        with self.assertRaises(ValueError):validate_plan(p)

    def test_source_age_and_command_caps_cannot_be_relaxed(self):
        p=self.plan('delivery_delay');validate_plan(p)
        p['command_lease_s']=.3
        with self.assertRaises(ValueError):validate_plan(p)

    def test_reply_hold_cannot_be_relabeled_as_natural_latency(self):
        p=self.plan('delivery_delay');p['source_perturbation']['reply_delivery_fault']['natural_service_delay']=True
        with self.assertRaises(ValueError):validate_plan(p)

    def test_small_white_patches_do_not_satisfy_broad_band_descriptor(self):
        pixels=glint_pixels();evidence=band_evidence(pixels,[0.,0.,64.,64.])
        self.assertEqual(evidence['status'],'REJECTED')

    def audit_fixture(self):
        trace=[];terminal=[]
        for i,hold in enumerate((3.,9.)):
            source=100.+i*20.;binding=dict(request_id=str(i),readback_receipt_monotonic_s=source)
            trace.append(dict(request_binding=binding,backend_started_monotonic_s=source+.1,
                backend_completed_monotonic_s=source+1.,reply_completed_monotonic_s=source+1.,
                controlled_reply_delivery=dict(request_binding=binding,original_reply_binding=binding,
                    declared_hold_s=hold,original_reply_completed_monotonic_s=source+1.,
                    natural_service_delay=False,service_reply_clock_and_content_unchanged=True,
                    delivery_status='DELIVERED',hold_started_monotonic_s=source+1.1,
                    hold_completed_monotonic_s=source+1.1+hold,delivered_to_policy_monotonic_s=source+1.1+hold)))
            terminal.append(dict(request_binding=binding,status='REJECTED' if i else 'ACCEPTED_CHAIN',
                reason='RESPONSE_TIMEOUT' if i else 'SOURCE_TO_CURRENT_CHAIN_ACCEPTED',transport_done=True))
        report=dict(configuration=dict(source_perturbation=specification('delivery_delay','SHORT_XY')),
            epoch_monotonic_s=99.,observer_close=dict(request_terminal_trace=dict(terminal_request_trace=terminal)))
        return report,trace

    def test_expired_real_reply_must_be_rejected(self):
        report,trace=self.audit_fixture()
        self.assertTrue(delivery_audit(report,trace)['status'].startswith('PASS'))
        report['observer_close']['request_terminal_trace']['terminal_request_trace'][1]['status']='ACCEPTED_CHAIN'
        self.assertTrue(delivery_audit(report,trace)['status'].startswith('FAIL'))

    def test_auditor_rejects_rebased_service_completion(self):
        report,trace=self.audit_fixture()
        trace[1]['controlled_reply_delivery']['original_reply_completed_monotonic_s']+=9.
        self.assertTrue(delivery_audit(report,trace)['status'].startswith('FAIL'))

    def test_actual_service_and_injected_hold_are_counted_separately(self):
        report,trace=self.audit_fixture();r=delivery_audit(report,trace)
        self.assertEqual(r['controlled_delays_count'],2)
        self.assertEqual(r['natural_service_over_two_s_count'],0)


if __name__=='__main__':unittest.main()

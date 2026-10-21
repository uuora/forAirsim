"""Evidence regressions: endpoints, missing slots, idle receipts and failed starts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from analyze_t1_motion_scale import (endpoint_displacement, motion_scale_evidence,
    measured_contract, continuous_visibility, motion_qualification, handoff_evidence_v6,
    command_responses, idle_omission_evidence, preflight_failure)
from test_t1_motion_plan_contract import plan


def endpoints(dx=.025, dy=0.):
    return dict(low_speed_stage_start_state=dict(host_receipt_monotonic_s=100.,
            position_world_ned_m=[0.,-3.,-3.],velocity_ned_m_s=[0.,0.,0.]),
        low_speed_stage_end_state=dict(host_receipt_monotonic_s=120.5,
            position_world_ned_m=[dx,-3.+dy,-3.],velocity_ned_m_s=[0.,0.,0.]))


def complete_report(p):
    slots=[];frames=[]
    for s in p['planned_slots']:
        i=s['slot_index'];fid=f'f{i}'
        slots.append(dict(slot_index=i,phase_id='visible',frame_id=fid,status='CAPTURED',
            offset_s=s['scheduled_start_s'],deadline_offset_s=s['scheduled_end_s'],
            due_monotonic_s=100+s['scheduled_start_s'],deadline_monotonic_s=100+s['scheduled_end_s']))
        frames.append(dict(slot_index=i,frame_id=fid,phase_id='visible',experiment_scope='MEASURED_RESEARCH',
            capture_started_host_s=100+s['scheduled_start_s']+.02,capture_received_host_s=100+s['scheduled_start_s']+.03))
    return dict(control_architecture=p['control_architecture'],epoch_monotonic_s=100.,
        measurement_entered=True,phase_transitions=[dict(phase_id='visible',started_monotonic_s=99.8,completed_monotonic_s=99.9)],
        frames=frames,slots=slots,acquisition_slots=[{}]*50,acquisition_frames=[{}]*6)


class MotionScaleTests(unittest.TestCase):
    def test_endpoint_pass_and_path_is_context_only(self):
        r=motion_scale_evidence(plan(),endpoints(.025),1,sampled_path=.001)
        self.assertEqual(r['status'],'PASS_ENDPOINT_MOTION_SCALE')
        self.assertFalse(r['path_used_for_qualification'])

    def test_large_path_or_takeoff_cannot_replace_zero_net_endpoint(self):
        r=endpoints(0);r['takeoff_displacement_m']=3.
        self.assertEqual(motion_scale_evidence(plan(),r,30,.15)['status'],
                         'FAIL_ENDPOINT_MOTION_SCALE_NOT_REACHED')

    def test_missing_reversed_nonfinite_or_boolean_endpoint_is_unknown(self):
        for case in ('missing','clock','nan','boolean'):
            with self.subTest(case=case):
                r=endpoints()
                if case=='missing':r.pop('low_speed_stage_end_state')
                elif case=='clock':r['low_speed_stage_end_state']['host_receipt_monotonic_s']=99.
                elif case=='nan':r['low_speed_stage_end_state']['position_world_ned_m'][0]=float('nan')
                else:r['low_speed_stage_start_state']['position_world_ned_m'][0]=True
                self.assertEqual(endpoint_displacement(r)['status'],'UNVERIFIED_ENDPOINT')

    def test_zero_gate_requires_no_nonzero_even_with_no_drift(self):
        p=plan('ZERO_GATE')
        self.assertEqual(motion_scale_evidence(p,endpoints(0),0)['status'],'PASS_ZERO_GATE_DRIFT')
        self.assertEqual(motion_scale_evidence(p,endpoints(0),1)['status'],'FAIL_ZERO_GATE_DRIFT_OR_NONZERO')

    def test_threshold_mutation_fails(self):
        p=plan();p['motion_scale_validation']['minimum_endpoint_net_xy_displacement_m']=.002
        self.assertEqual(motion_scale_evidence(p,endpoints(.025),1)['status'],'FAIL_FROZEN_THRESHOLD_CONTRACT')

    def test_full_grid_and_separate_acquisition(self):
        p=plan();r=complete_report(p);c=measured_contract(p,r)
        self.assertEqual(c['status'],'PASS_COMPLETE_VISIBLE_GRID')
        self.assertEqual(c['saved_frames'],50);self.assertEqual(c['acquisition_frames_preserved'],6)

    def test_missing_duplicate_shifted_and_declared_41_fail(self):
        for case in ('missing','duplicate','clock','stale_count','hidden'):
            with self.subTest(case=case):
                p=plan();r=complete_report(p)
                if case=='missing':r['frames'].pop();r['slots'][-1]['status']='MISSED'
                elif case=='duplicate':r['frames'][-1]=copy.deepcopy(r['frames'][-2])
                elif case=='clock':r['slots'][8]['due_monotonic_s']+=.1
                elif case=='stale_count':p['motion_scale_validation']['planned_measured_slots']=41
                else:r['phase_transitions'][0]['phase_id']='hidden'
                self.assertEqual(measured_contract(p,r)['status'],'FAIL_MEASURED_CONTRACT')

    def test_visibility_missing_grid_or_mask_stays_unknown(self):
        rows=[dict(frame_id='f1',status='SAVED_MASK_VERIFIED',target_visible_pixels=501)]
        c=dict(status='PASS_COMPLETE_VISIBLE_GRID',expected_slots=1)
        self.assertEqual(continuous_visibility(rows,c)['status'],'PASS_SAMPLED_VISIBLE_COMPLETE_GRID')
        self.assertEqual(continuous_visibility(rows,dict(c,expected_slots=2))['status'],'UNVERIFIED_CONTINUOUS_VISIBLE_GRID')
        rows[0]['target_visible_pixels']=499
        self.assertEqual(continuous_visibility(rows,c)['status'],'UNVERIFIED_CONTINUOUS_VISIBLE_GRID')

    def test_motion_cannot_override_missing_handoff(self):
        report=complete_report(plan())
        # Real auditor requires a handoff receipt in every saved frame.
        report['acquisition_frames']=[]
        for i,row in enumerate(report['frames']):row['frame_sequence']=i
        handoff=handoff_evidence_v6(report)
        self.assertTrue(handoff['issues'])
        self.assertEqual(motion_qualification(dict(independent_integration_qualification='PASS_CONTROLLED_DEVELOPMENT_SCOPE',
            review_handoff=handoff)),'FAIL_V6_HANDOFF_EVIDENCE')

    def test_dispatch_freshness_and_query_cadence_are_distinct(self):
        report=endpoints();report['frames']=[dict(frame_id='f1',command_self_state=dict(host_receipt_monotonic_s=10.,
            position_world_ned_m=[0.,0.,-3.],velocity_ned_m_s=[0.,0.,0.]),motion_gate={},lease_stop={})]
        cmd=dict(target_frame_id='f1',command_id='c1',dispatch_monotonic_s=10.01,velocity_xy_m_s=[.01,0.],duration_s=.12,
                 lease_clear=dict(actual_zero_clear_s=10.15,declared_lease_end_s=10.13))
        run=dict(target_command_rows=[cmd]);events=[dict(cmd,kind='COMMAND',stage='PERCEPTION_XY')]
        out=command_responses(report,run,events)
        self.assertTrue(out['rows'][0]['self_state_fresh_at_dispatch_verified'])
        report['frames'][0]['command_self_state']['host_receipt_monotonic_s']=9.8
        self.assertFalse(command_responses(report,run,events)['rows'][0]['self_state_fresh_at_dispatch_verified'])

    def test_idle_requires_real_zero_and_fresh_stop_samples(self):
        events=[dict(kind='COMMAND',command_id='z',dispatch_monotonic_s=10.,velocity_xy_m_s=[0.,0.])]
        events += [dict(kind='SELF_STATE',host_receipt_monotonic_s=t,velocity_ned_m_s=[0.,0.,0.]) for t in (10.04,10.1)]
        events.append(dict(kind='IDLE_ZERO_OMITTED',decision_monotonic_s=10.11,self_state_receipt_monotonic_s=10.1,
            actual_velocity_xy_m_s=[0.,0.],last_actual_zero_dispatch_monotonic_s=10.,sampled_stop_started_monotonic_s=10.04,
            sampled_stop_clear_monotonic_s=10.,distinct_post_clear_sample_count=2,configured_real_zero_refresh_s=.30,
            command_dispatched=False,command_receipt_created=False,active_target_lease=False,target_observation_used=False))
        response=dict(active_intervals_actual_dispatch_to_explicit_clear=[]);scheduling=dict(idle_zero_omission_count=1)
        self.assertEqual(idle_omission_evidence(events,response,scheduling)['issues'],[])
        for case in ('invented_command','expired','moving','active','no_zero'):
            with self.subTest(case=case):
                e=copy.deepcopy(events);r=copy.deepcopy(response)
                if case=='invented_command':e[-1]['command_receipt_created']=True
                elif case=='expired':e[-1]['decision_monotonic_s']=10.31
                elif case=='moving':e[-1]['actual_velocity_xy_m_s']=[.01,0.]
                elif case=='active':r['active_intervals_actual_dispatch_to_explicit_clear']=[dict(start_s=10.05,end_s=10.2)]
                else:e[0]['velocity_xy_m_s']=[.002,0.]
                self.assertTrue(idle_omission_evidence(e,r,scheduling)['issues'])

    def test_failed_preflight_has_no_imputed_flight_measurements(self):
        from analyze_t1_identity_flight import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'plan.json';p.write_text(json.dumps(plan()),encoding='utf-8')
            manifest=dict(plan=str(p),plan_sha256=digest(p),runner_started=False,status='FAIL',session_id='failed')
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            (root/'plan_preflight.json').write_text(json.dumps(dict(status='FAIL_MOTION_PLAN_CONTRACT',
                simulator_calls=False,plan_sha256=digest(p))),encoding='utf-8')
            out=preflight_failure(root,manifest)
            self.assertEqual(out['actual_flights'],0);self.assertIsNone(out['nonzero_target_command_count'])
            self.assertFalse(out['flight_measurements_imputed'])

    def test_later_dispatch_with_same_clock_does_not_make_idle_active(self):
        events=[dict(kind='COMMAND',command_id='z',dispatch_monotonic_s=10.,velocity_xy_m_s=[0.,0.])]
        events += [dict(kind='SELF_STATE',host_receipt_monotonic_s=t,velocity_ned_m_s=[0.,0.,0.]) for t in (10.04,10.1)]
        events.append(dict(kind='IDLE_ZERO_OMITTED',decision_monotonic_s=10.11,self_state_receipt_monotonic_s=10.1,
            actual_velocity_xy_m_s=[0.,0.],last_actual_zero_dispatch_monotonic_s=10.,sampled_stop_started_monotonic_s=10.04,
            sampled_stop_clear_monotonic_s=10.,distinct_post_clear_sample_count=2,configured_real_zero_refresh_s=.30,
            command_dispatched=False,command_receipt_created=False,active_target_lease=False,target_observation_used=False))
        events += [dict(kind='COMMAND',command_id='target',dispatch_monotonic_s=10.11,velocity_xy_m_s=[.01,0.]),
                   dict(kind='COMMAND',command_id='clear',dispatch_monotonic_s=10.24,velocity_xy_m_s=[0.,0.])]
        response=dict(active_intervals_actual_dispatch_to_explicit_clear=[dict(start_s=10.11,end_s=10.24,
            command_id='target',dispatch_event_index=4,clear_event_index=5)])
        result=idle_omission_evidence(events,response,dict(idle_zero_omission_count=1))
        self.assertEqual(result['issues'],[])
        self.assertEqual(result['rows'][0]['same_clock_later_dispatch_ids'],['target'])
        # If an actual target dispatch precedes this omission, the audit fails.
        response['active_intervals_actual_dispatch_to_explicit_clear'][0]['dispatch_event_index']=2
        self.assertIn('OMISSION_DURING_RECORDED_ACTIVE_TARGET',
            idle_omission_evidence(events,response,dict(idle_zero_omission_count=1))['issues'])

    def test_absent_report_alone_is_not_zero_flights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'plan.json';p.write_text(json.dumps(plan()),encoding='utf-8')
            manifest=dict(plan=str(p),status='FAIL',session_id='unknown')
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            self.assertIsNone(preflight_failure(root,manifest)['actual_flights'])


if __name__=='__main__':unittest.main()

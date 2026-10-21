"""Evidence boundary regressions; no simulator or model calls."""
import copy
import unittest

from analyze_t1_v6p1_late_residual import frame_stages, post_clear_motion


def timing_report(clear=10.39, stopped=10.46, processing=10.30):
    f=dict(frame_id='f',slot_index=0,phase_id='visible',due_monotonic_s=10.,
        deadline_monotonic_s=10.4,capture_started_host_s=10.,capture_received_host_s=10.09,
        observer_result=dict(process_completed_host_monotonic_s=10.22),
        motion_gate=dict(decision_monotonic_s=10.26),motion_decision_available_monotonic_s=10.27,
        command=dict(dispatch_monotonic_s=10.26),frame_processing_completed_monotonic_s=processing,
        lease_stop=dict(declared_lease_end_monotonic_s=10.38,
            target_setpoint_cleared_monotonic_s=clear,completed_monotonic_s=stopped),
        decision_completed_monotonic_s=max(stopped,processing))
    return dict(frames=[f],slots=[dict(slot_index=0,phase_id='visible',status='CAPTURED',deadline_monotonic_s=10.4)])


def residual_fixture(times=(10.,10.1,10.12,10.16,10.22,10.32,10.4,10.5,11.)):
    def state(t):
        return dict(kind='SELF_STATE',stage='PERCEPTION_XY',host_receipt_monotonic_s=t,
            position_world_ned_m=[0.,(t-10.)*.001,-3.],velocity_ned_m_s=[0.,.001,0.])
    r=dict(low_speed_stage_start_state=state(10.),low_speed_stage_end_state=state(11.),
        frames=[dict(frame_id='f',lease_stop=dict(completed_monotonic_s=10.20))])
    commands=[dict(kind='COMMAND',stage='PERCEPTION_XY',command_id='c',dispatch_monotonic_s=10.,velocity_xy_m_s=[0.,.01]),
        dict(kind='COMMAND',stage='PERCEPTION_XY',command_id='z',dispatch_monotonic_s=10.12,velocity_xy_m_s=[0.,0.]),
        dict(kind='COMMAND',stage='PERCEPTION_XY',command_id='other',dispatch_monotonic_s=10.4,velocity_xy_m_s=[.01,0.])]
    events=commands+[state(t) for t in times]
    base=dict(command_response=dict(active_intervals_actual_dispatch_to_explicit_clear=[dict(
        command_id='c',start_s=10.,end_s=10.12,dispatch_event_index=0,clear_event_index=1)]),
        target_command_rows=[dict(command_id='c',target_frame_id='f',lease_clear=dict(issues=[]))])
    return r,events,base


class StageTests(unittest.TestCase):
    def test_post_clear_wait_crossing_and_overlap_not_sum(self):
        t=frame_stages(timing_report());r=t['rows'][0]
        self.assertEqual(r['attribution'],'POST_CLEAR_STOP_RECEIPT_CROSSED_DEADLINE')
        self.assertAlmostEqual(r['slot_to_full_s'],.46)
        self.assertAlmostEqual(r['critical_tail_after_processing_s'],.16)

    def test_clear_already_late(self):
        self.assertEqual(frame_stages(timing_report(clear=10.42))['rows'][0]['attribution'],'CLEAR_ALREADY_LATE')

    def test_deadline_equality_is_timely(self):
        self.assertEqual(frame_stages(timing_report(stopped=10.4))['rows'][0]['attribution'],'TIMELY')

    def test_processing_late_and_both_late_are_separate(self):
        self.assertEqual(frame_stages(timing_report(stopped=10.4,processing=10.45))['rows'][0]['attribution'],'FRAME_PROCESSING_LATE')
        self.assertEqual(frame_stages(timing_report(processing=10.48))['rows'][0]['attribution'],'BOTH_BRANCHES_LATE')

    def test_missing_or_wrong_final_clock_cannot_mint_attribution(self):
        for mutation in ('missing','wrong'):
            r=timing_report()
            if mutation=='missing':r['frames'][0]['lease_stop'].pop('completed_monotonic_s')
            else:r['frames'][0]['decision_completed_monotonic_s']+=.01
            with self.assertRaises(ValueError):frame_stages(r)


class ResidualTests(unittest.TestCase):
    def test_strict_clear_boundary_and_next_any_nonzero_not_included(self):
        out=post_clear_motion(*residual_fixture())
        p=out['categories']['POST_CLEAR_INTERIOR']
        self.assertEqual(p['segment_count'],2)
        self.assertAlmostEqual(p['displacement_vector_norm_m'],.00016)
        self.assertAlmostEqual(out['after_stop_receipt_motion']['displacement_vector_norm_m'],.00010)
        self.assertEqual(out['windows'][0]['next_nonzero_command_id'],'other')
        self.assertEqual(out['endpoint_reconciliation_error_m'],0.)
        self.assertTrue(all(s['left_receipt_s']>10.12 and s['right_receipt_s']<10.4
            for s in out['segments'] if s['category']=='POST_CLEAR_INTERIOR'))

    def test_no_interior_samples_remain_unknown(self):
        out=post_clear_motion(*residual_fixture(times=(10.,10.12,10.4,11.)))
        self.assertEqual(out['covered_window_count'],0)
        self.assertIsNone(out['windows'][0]['sampled_motion']['displacement_vector_norm_m'])

    def test_tail_ends_at_explicit_research_endpoint(self):
        r,e,b=residual_fixture();e=[x for x in e if x.get('command_id')!='other']
        out=post_clear_motion(r,e,b)
        self.assertEqual(out['windows'][0]['end_kind'],'EXPLICIT_STAGE_END')
        self.assertEqual(out['windows'][0]['last_interior_receipt_s'],11.)
        self.assertAlmostEqual(out['categories']['POST_CLEAR_INTERIOR']['displacement_vector_norm_m'],.00084)

    def test_zero_output_control_has_no_target_residual_population(self):
        r,e,b=residual_fixture();b['target_command_rows']=[];b['command_response']['active_intervals_actual_dispatch_to_explicit_clear']=[]
        out=post_clear_motion(r,e,b)
        self.assertEqual(out['after_target_clear_status'],'NOT_APPLICABLE_NO_NONZERO_TARGET')
        self.assertEqual(out['window_count'],0)
        self.assertIsNone(out['categories']['POST_CLEAR_INTERIOR']['displacement_vector_norm_m'])

    def test_conflicting_same_host_tick_position_remains_unallocated(self):
        r,e,b=residual_fixture();bad=copy.deepcopy(e[-2]);bad['position_world_ned_m'][0]=.1;e.append(bad)
        out=post_clear_motion(r,e,b)
        self.assertEqual(out['ambiguous_host_tick_count'],1)
        self.assertEqual(out['categories']['AMBIGUOUS_HOST_TICK_UNALLOCATED']['segment_count'],2)
        self.assertEqual(len(out['ambiguous_host_tick_records'][0]['records']),2)
        self.assertEqual(out['endpoint_reconciliation_error_m'],0.)

    def test_missing_explicit_zero_event_match_refuses(self):
        r,e,b=residual_fixture();b['command_response']['active_intervals_actual_dispatch_to_explicit_clear'][0]['clear_event_index']=None
        with self.assertRaisesRegex(ValueError,'UNMATCHED_EXPLICIT_ZERO'):post_clear_motion(r,e,b)


if __name__=='__main__':unittest.main()

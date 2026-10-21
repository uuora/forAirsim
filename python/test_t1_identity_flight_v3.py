"""Pure phase, current-acquisition and denominator tests; no simulator or API."""
import copy
import math
import unittest

from t1_identity_flight_v3 import (
    acquisition_slots, acquisition_established, close_acquisition_tail,
    finalize_unresolved_slots, frame_row, planned_slots,
)
from prepare_t1_identity_flight_v3 import NEW_SOURCES


class InitialIdentityAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(period_s=.4,
            initial_identity_acquisition=dict(maximum_duration_s=20.,period_s=.4),
            phases=[dict(phase_id='visible',duration_s=7.),
                    dict(phase_id='hidden',duration_s=2.),dict(phase_id='return',duration_s=7.)])
        self.valid=dict(status='FRAME_COMPLETED',
            observer_result=dict(semantic_frame=dict(readback_receipt_monotonic_s=19.5),
                temporal_identity=dict(status='TEMPORAL_LOCAL_ASSOCIATION_VALID',state='TRACKING')),
            surface=dict(status='SURFACE_GEOMETRY_ACCEPTED'))

    def test_initial_grid_has_all_fifty_slots_and_twenty_second_cap(self):
        slots,total=acquisition_slots(self.plan)
        self.assertEqual((len(slots),total),(50,20.))
        self.assertAlmostEqual(slots[-1]['offset_s'],19.6)
        self.assertEqual(slots[-1]['deadline_offset_s'],20.)
        self.assertEqual([s['slot_index'] for s in slots],list(range(50)))
        self.assertTrue(all(s['experiment_scope']=='INITIAL_IDENTITY_ACQUISITION' for s in slots))

    def test_measured_phase_grid_is_unchanged_and_boundary_clipped(self):
        slots,total=planned_slots(self.plan)
        self.assertEqual((len(slots),total),(41,16.))
        self.assertEqual(slots[17]['deadline_offset_s'],7.)
        self.assertEqual(slots[18]['offset_s'],7.)
        self.assertEqual(slots[22]['deadline_offset_s'],9.)
        zero=copy.deepcopy(self.plan);zero['phases']=[dict(phase_id='visible',duration_s=8.)]
        self.assertEqual(tuple(map(lambda x:len(x) if isinstance(x,list) else x,planned_slots(zero))),(20,8.))

    def test_reject_changed_acquisition_grid(self):
        for key,value in [('maximum_duration_s',21.),('period_s',.5)]:
            with self.subTest(key=key):
                plan=copy.deepcopy(self.plan);plan['initial_identity_acquisition'][key]=value
                with self.assertRaises(ValueError):acquisition_slots(plan)

    def test_current_tracking_and_surface_succeed_before_global_deadline(self):
        self.assertTrue(acquisition_established(self.valid,now=19.999,deadline=20.))
        self.assertFalse(acquisition_established(self.valid,now=20.,deadline=20.))
        self.assertFalse(acquisition_established(self.valid,now=20.001,deadline=20.))

    def test_pending_identity_or_rejected_surface_cannot_start_measurement(self):
        for path,value in [('state','RECOVERY_PENDING'),('status','NO_TARGET_OUTPUT')]:
            row=copy.deepcopy(self.valid);row['observer_result']['temporal_identity'][path]=value
            self.assertFalse(acquisition_established(row,now=19.8,deadline=20.))
        row=copy.deepcopy(self.valid);row['surface']['status']='SURFACE_REJECTED'
        self.assertFalse(acquisition_established(row,now=19.8,deadline=20.))
        row=copy.deepcopy(self.valid);row['status']='FRAME_STARTED'
        self.assertFalse(acquisition_established(row,now=19.8,deadline=20.))

    def test_current_frame_freshness_and_finite_clock_are_still_required(self):
        for receipt in [19.,20.1,float('nan'),True,None]:
            row=copy.deepcopy(self.valid);row['observer_result']['semantic_frame']['readback_receipt_monotonic_s']=receipt
            self.assertFalse(acquisition_established(row,now=19.8,deadline=20.))
        for value in [float('nan'),float('inf'),True]:
            self.assertFalse(acquisition_established(self.valid,now=value,deadline=20.))

    def test_global_ids_continue_after_consumed_acquisition_slots(self):
        initial,_=acquisition_slots(self.plan)
        initial[0]['status']='CAPTURED';initial[1]['status']='NOT_CAPTURED_SLOT_DEADLINE'
        initial[2]['status']='CAPTURED'
        base=close_acquisition_tail(initial,2,1.2)
        self.assertEqual(base,3)
        self.assertTrue(all(s['status']=='NOT_REQUIRED_AFTER_ACQUISITION' for s in initial[3:]))
        main,_=planned_slots(self.plan)
        rows=[frame_row(initial[i],'trial',0,acquisition=True) for i in (0,2)]
        rows += [frame_row(main[i],'trial',base) for i in (0,1,3)]
        self.assertEqual([r['frame_sequence'] for r in rows],[0,2,3,4,6])
        self.assertEqual(len({r['frame_id'] for r in rows}),5)
        self.assertEqual(rows[2]['frame_id'],'trial_f003')
        self.assertEqual(rows[2]['slot_index'],0)
        self.assertEqual(rows[2]['experiment_scope'],'MEASURED_RESEARCH')

    def test_completed_tail_keeps_all_missing_acquisition_prefix_slots(self):
        slots,_=acquisition_slots(self.plan)
        slots[0]['status']='NOT_CAPTURED_SLOT_DEADLINE';slots[1]['status']='CAPTURED'
        close_acquisition_tail(slots,1,.8)
        self.assertEqual(len(slots),50)
        self.assertEqual(slots[0]['status'],'NOT_CAPTURED_SLOT_DEADLINE')
        self.assertEqual(slots[1]['status'],'CAPTURED')

    def test_failed_acquisition_preserves_all_main_slots_outside_measured_frames(self):
        initial,_=acquisition_slots(self.plan);main,_=planned_slots(self.plan)
        frames=[dict(slot_index=0,media=dict(rgb=dict(path='retained_rgb')),status='FRAME_STARTED')]
        finalize_unresolved_slots(initial,frames)
        finalize_unresolved_slots(main,[],main_not_entered=True)
        self.assertEqual(initial[0]['status'],'CAPTURED_PROCESSING_FAILED')
        self.assertEqual(len(main),41)
        self.assertTrue(all(s['status']=='NOT_ENTERED_INITIAL_ACQUISITION' for s in main))
        self.assertFalse(any('due_monotonic_s' in s for s in main))

    def test_global_sequence_rejects_negative_or_bool_indices(self):
        for base,index in [(-1,0),(True,0),(0,-1),(0,True)]:
            with self.assertRaises(ValueError):frame_row(dict(slot_index=index),'trial',base)

    def test_source_inventory_is_explicit_and_excludes_credentials(self):
        self.assertEqual(len(NEW_SOURCES),7)
        self.assertIn('python/t1_delayed_semantic_bridge.py',NEW_SOURCES)
        self.assertIn('python/t1_candidate_surface_v3.py',NEW_SOURCES)
        self.assertFalse(any('.env' in n or 'credential' in n for n in NEW_SOURCES))


if __name__=='__main__':unittest.main()

"""Synthetic timing/scene guard regression. Never flight evidence."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import Future
import numpy as np
from test_fixed_map_perception_execution import row,geometry,CFG,SAFETY
from fixed_map_perception_execution import PerceptionExecutor,TrackingMarginGeometry
from fixed_map_perception_task import build_preview,live_estimate
from fixed_map_random_scene import sample,REGION
from fixed_map_perception import read,HIST
class HandoffTests(unittest.TestCase):
    def owner(self,folder):
        e=PerceptionExecutor.__new__(PerceptionExecutor);e.service=SimpleNamespace(latest=row(),cfg=dict(CFG,plan_fresh_handoff_timeout_s=2.));e.p=np.array([0.,0.,3.1]);e.static_geometry=geometry();e.safety=SAFETY;e.stage='ARRIVED';e.plan_version=0;e.reason='';e.resume_required=False;e.run_dir=Path(folder);e.events=[];e.event=lambda k,**v:e.events.append((k,v))
        f=Future();f.set_result(build_preview(row(),[0,0,3.1],e.static_geometry,SAFETY,CFG,10.1));e.pending_plan=dict(future=f,execute=False,actor='synthetic',submitted_monotonic_s=10.1)
        return e
    def test_expired_snapshot_waits_new_frame_without_extending_old_age(self):
        with tempfile.TemporaryDirectory() as f:
            e=self.owner(f)
            with patch('fixed_map_perception_execution.time.monotonic',return_value=10.6):e.poll_plan()
            self.assertEqual(e.preview_result['status'],'WAITING_FRESH_REVALIDATION');self.assertIsNotNone(e.pending_plan)
            e.service.latest=row(receipt=10.65,frame='new-2')
            with patch('fixed_map_perception_execution.time.monotonic',return_value=10.75):e.poll_plan()
            self.assertEqual(e.preview_result['status'],'FRESH_REVALIDATED_PERCEPTION_PLAN');self.assertEqual(e.preview_result['estimate']['frame_id'],'new-2');self.assertEqual(e.stage,'ARRIVED');self.assertFalse(any(k=='PERCEPTION_START_ACCEPTED' for k,v in e.events))
    def test_handoff_timeout_keeps_rejection(self):
        with tempfile.TemporaryDirectory() as f:
            e=self.owner(f)
            with patch('fixed_map_perception_execution.time.monotonic',return_value=10.6):e.poll_plan()
            with patch('fixed_map_perception_execution.time.monotonic',return_value=12.7):e.poll_plan()
            self.assertEqual(e.preview_result['status'],'REJECTED');self.assertIsNone(e.pending_plan)
    def test_scene_version_bound_and_rejected_when_changed(self):
        r=row();cfg=dict(CFG,runtime_scene_version='scene-2')
        with self.assertRaisesRegex(ValueError,'SCENE_VERSION'):live_estimate(r,cfg,10.1)
        r['capture']['runtime_scene_version']=r['position_estimate']['runtime_scene_version']='scene-2';self.assertTrue(live_estimate(r,cfg,10.1)['valid'])
    def test_random_samples_bounded_reproducible_and_map_not_input_to_task(self):
        parts=read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives']
        a,_=sample(24101001,parts);b,_=sample(24101001,parts);self.assertTrue(np.array_equal(a,b));self.assertGreaterEqual(a[0],42);self.assertLessEqual(a[0],49);self.assertTrue(-7<=a[1]<=-1.5)
    def test_margin_increases_planning_threshold(self):
        g=TrackingMarginGeometry(geometry().parts,.75,.15)
        with patch('fixed_map_mission.Geometry.plan',return_value=[]) as p:g.plan([0,0,3.1],[10,0,3.1],1.25)
        self.assertAlmostEqual(p.call_args.args[2],1.4)
if __name__=='__main__':unittest.main()

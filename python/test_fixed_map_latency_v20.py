import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import test_fixed_map_perception_execution as fixtures
from fixed_map_perception import expired_queue_report
from fixed_map_perception_task import live_estimate
from fixed_map_resources import ResourceTrace


class QueueExpiryTests(unittest.TestCase):
    def test_fresh_boundary_is_not_rejected(self):
        self.assertIsNone(expired_queue_report(fixtures.row()['capture'],fixtures.CFG,10.5))
    def test_expired_input_retains_original_binding_and_refuses_motion(self):
        meta=fixtures.row()['capture'];row=expired_queue_report(meta,fixtures.CFG,10.501)
        self.assertFalse(row['valid']);self.assertEqual(row['capture']['received_monotonic_s'],10.)
        self.assertEqual(row['position_estimate']['image_sha256'],meta['image_sha256'])
        self.assertTrue(row['queue_rejected_before_inference']);self.assertEqual(row['inference_wall_s'],0.)
        with self.assertRaises(ValueError):live_estimate(row,fixtures.CFG,10.501)
    def test_future_source_is_not_made_valid(self):
        self.assertFalse(expired_queue_report(fixtures.row()['capture'],fixtures.CFG,9.9)['valid'])


class ResourceTraceTests(unittest.TestCase):
    def test_successful_observer_has_no_control_actions(self):
        with tempfile.TemporaryDirectory() as path:
            trace=ResourceTrace(Path(path)/'trace.jsonl',lambda:dict(stage='GROUND'),lambda:{})
            with patch('fixed_map_resources.memory_headroom',return_value=dict(physical_available_bytes=4,commit_available_bytes=3)):
                trace.start();trace.thread.join(timeout=.03);receipt=trace.stop()
            self.assertEqual(receipt['status'],'PASS');self.assertGreaterEqual(receipt['samples'],1)
    def test_observer_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as path:
            trace=ResourceTrace(Path(path)/'trace.jsonl',lambda:{},lambda:{})
            with patch('fixed_map_resources.memory_headroom',side_effect=OSError('memory unavailable')):
                trace.start();trace.thread.join(timeout=1);receipt=trace.stop()
            self.assertEqual(receipt['status'],'FAIL');self.assertIn('memory unavailable',receipt['errors'])
    def test_observer_rate_is_bounded(self):
        for period in [0,.1,6]:
            with self.assertRaises(ValueError):ResourceTrace('unused',lambda:{},lambda:{},period)


if __name__=='__main__':unittest.main()

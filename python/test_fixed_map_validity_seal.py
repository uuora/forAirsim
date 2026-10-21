import copy
import unittest
import test_fixed_map_perception_execution as fixtures
from fixed_map_perception import seal_current_frame_report
from fixed_map_perception_task import live_estimate


class ValiditySealTests(unittest.TestCase):
    def report(self):
        row=fixtures.row();row.update(rejection_reasons=[],planning_preview=dict(status='VALID_ESTIMATE_AWAITING_EXPLICIT_PREVIEW'));return row
    def test_expiry_between_early_check_and_final_record_fails_closed(self):
        row=self.report();sealed=seal_current_frame_report(row,fixtures.CFG,10.501)
        self.assertFalse(sealed['valid']);self.assertFalse(sealed['position_estimate']['valid']);self.assertIn('CURRENT_FRAME_EXPIRED',sealed['rejection_reasons'])
        with self.assertRaises(ValueError):live_estimate(sealed,fixtures.CFG,sealed['processed_monotonic_s'])
    def test_fresh_final_record_retains_binding_and_timestamp(self):
        row=self.report();original=copy.deepcopy(row['position_estimate']);sealed=seal_current_frame_report(row,fixtures.CFG,10.49)
        self.assertTrue(sealed['valid']);self.assertEqual(sealed['validity_seal']['checked_monotonic_s'],sealed['processed_monotonic_s']);self.assertEqual(live_estimate(sealed,fixtures.CFG,10.49),original)
    def test_invalid_source_cannot_be_resurrected_by_a_new_timestamp(self):
        row=self.report();row['valid']=False
        self.assertFalse(seal_current_frame_report(row,fixtures.CFG,10.1)['position_estimate']['valid'])


if __name__=='__main__':unittest.main()

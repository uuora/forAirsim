"""Explicit timestamps/order of perception and truth evaluation remain separate."""
import ast
from pathlib import Path
import unittest

from prepare_t1_identity_sequence import build_protocol
from t1_identity_sequence_probe_v1p2 import planned_slots,observer_observation


class RGBFirstTests(unittest.TestCase):
    def test_perception_finishes_before_truth_access(self):
        text=Path(__file__).with_name("t1_identity_sequence_probe_v1p2.py").read_text(encoding="utf-8")
        start=text.index("            image,ranges,seg,meta=capture(client)")
        end=text.index('        wait_until(report["schedule_end_host_monotonic_s"])',start)
        body=text[start:end]
        self.assertLess(body.index("decision=observer.process"),body.index("truth={key:visible_mask"))
        self.assertLess(body.index("completed=time.monotonic()"),body.index("np.save(paths"))
        self.assertIn('"process_completed_host_monotonic_s":completed',body)
        self.assertIn("IMWRITE_PNG_COMPRESSION,0",body)
        ast.parse(text)
    def test_same_registered_grid(self):
        for mode in ("VISUAL_ATTRIBUTE","UNRESOLVABLE_IDENTICAL"):
            rows=planned_slots(build_protocol("x",6100410,mode))
            self.assertEqual(len(rows),60)
            self.assertEqual(rows[-1]["planned_end_relative_s"],12)
    def test_boundary_function_is_same_contract(self):
        import t1_identity_sequence_probe_v1p1 as previous
        self.assertEqual(observer_observation.__code__.co_code,previous.observer_observation.__code__.co_code)


if __name__=="__main__":
    unittest.main()

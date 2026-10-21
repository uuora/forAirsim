"""Timing repair preserves the task/denominator and labels cached metadata."""
import ast
from pathlib import Path
import unittest

from prepare_t1_identity_sequence import build_protocol
from t1_identity_sequence_probe_v1p1 import observer_observation, planned_slots


class TimingRepairTests(unittest.TestCase):
    def test_no_per_frame_metadata_rpc(self):
        tree=ast.parse(Path(__file__).with_name("t1_identity_sequence_probe_v1p1.py").read_text(encoding="utf-8"))
        loops=[node for node in ast.walk(tree) if isinstance(node,ast.For)
               and isinstance(node.target,ast.Name) and node.target.id=="slot"
               and any(isinstance(item,ast.Call) and isinstance(item.func,ast.Name)
                       and item.func.id=="capture" for item in ast.walk(node))]
        self.assertEqual(len(loops),1)
        # Metadata queries remain in the phase-transition IF, not the repeated
        # stable-scene path. Inspect only unconditional statements in that loop.
        forbidden={"simGetCameraInfo","simGetObjectPose","ground_states"}
        calls=[]
        for statement in loops[0].body:
            if isinstance(statement,ast.If):
                continue
            for node in ast.walk(statement):
                if isinstance(node,ast.Call):
                    calls.append(node.func.attr if isinstance(node.func,ast.Attribute)
                                 else node.func.id if isinstance(node.func,ast.Name) else "")
        self.assertFalse(forbidden.intersection(calls))
    def test_identical_sixty_slot_grid(self):
        slots=planned_slots(build_protocol("x",6100410,"VISUAL_ATTRIBUTE"))
        self.assertEqual(len(slots),60)
        self.assertEqual(slots[-1]["planned_end_relative_s"],12.)
    def test_cached_ground_never_enters_observer(self):
        frame={"session_id":"s", "frame_id":"f", "frame_sequence":1, "description":"d",
            "capture_started_host_s":10., "capture_received_host_s":10.1,"image_size_wh":[100,80],
            "response_camera_poses":[{"position_local_ned_m":[0,0,0],"orientation_xyzw":[0,0,0,1],
                "render_readback_timestamp_ns":100}],"scene_intrinsics":{},
            "world_from_local_translation_ned_m":[0,0,0],"media":{"rgb":{"path":"x","sha256":"a"*64}},
            "evaluation_only":{"target_world_ned_m":[1,2,3]}}
        frame["phase_ground_readback"]={"states":{"DroneA":{"landed":True}}}
        observation=observer_observation(frame)
        self.assertNotIn("phase_ground_readback",observation)
        self.assertNotIn("evaluation_only",observation)


if __name__=="__main__":
    unittest.main()

"""Pure timing, observer-boundary and intended-scene contract regressions."""
import copy
import unittest

from prepare_t1_identity_sequence import build_protocol
from t1_identity_sequence_probe import observer_observation,planned_slots,scene_acceptance,validate_protocol,wait_until


class SequenceContractTests(unittest.TestCase):
    def protocol(self,observability="VISUAL_ATTRIBUTE"):
        return build_protocol("test",6100410,observability)

    def test_exact_predeclared_slot_denominator(self):
        rows=planned_slots(self.protocol())
        self.assertEqual(len(rows),60)
        self.assertEqual([sum(r["phase_id"]==p for r in rows) for p in ("initial_pair","target_hidden_with_distractor_at_old_target","return_target_right")],[25,10,25])
        self.assertEqual([r["frame_sequence"] for r in rows],list(range(1,61)))
        self.assertTrue(all(r["capture_status"]=="NOT_CAPTURED" for r in rows))
        self.assertAlmostEqual(rows[-1]["planned_end_relative_s"],12)
        for before,after in zip(rows,rows[1:]):
            self.assertAlmostEqual(before["planned_end_relative_s"],after["planned_start_relative_s"])

    def test_no_post_occlusion_location_hint(self):
        for observability in ("VISUAL_ATTRIBUTE","UNRESOLVABLE_IDENTICAL"):
            cfg=self.protocol(observability)
            validate_protocol(cfg)
            self.assertNotIn("右",cfg["description"])
            self.assertEqual(cfg["target_generation"],1)
        cfg=self.protocol()
        cfg["description"]="遮挡后选择右侧红气球"
        with self.assertRaises(ValueError):
            validate_protocol(cfg)

    def test_generation_never_phase_reset(self):
        cfg=self.protocol()
        cfg["target_generation"]=2
        with self.assertRaises(ValueError):
            validate_protocol(cfg)

    def test_fixed_host_duration_and_period_validation(self):
        for value in (.1,0,float("nan"),float("inf")):
            cfg=self.protocol();cfg["minimum_period_s"]=value
            with self.assertRaises(ValueError):
                planned_slots(cfg)
        cfg=self.protocol();cfg["phases"][1]["host_duration_s"]=3
        with self.assertRaises(ValueError):
            planned_slots(cfg)

    def test_observer_boundary_allowlist_and_deep_copy(self):
        frame={"session_id":"s","frame_id":"f","frame_sequence":3,"description":"d",
            "capture_started_host_s":10,"capture_received_host_s":10.1,"image_size_wh":[640,480],
            "response_camera_poses":[{"position_local_ned_m":[1,2,3],"orientation_xyzw":[0,0,0,1],"render_readback_timestamp_ns":99}],
            "scene_intrinsics":{"K_nominal":[[320,0,320],[0,320,240],[0,0,1]]},
            "world_from_local_translation_ned_m":[0,-3,-1],
            "media":{"rgb":{"path":"rgb.png","sha256":"a"*64},"segmentation":{"path":"seg.png"},"depth":{"path":"depth.npy"}},
            "phase_id":"return_target_right","evaluation_only":{"target_world_ned_m":[0,.8,-3],"target":{"visible_pixel_count":900}}}
        before=copy.deepcopy(frame)
        observation=observer_observation(frame)
        self.assertEqual(set(observation),{"session_id","target_generation","frame_id","frame_sequence","description","rgb_path","rgb_sha256","image_size_wh","readback_receipt_monotonic_s","capture_requested_monotonic_s","capture_completed_monotonic_s","camera","capture_metadata"})
        self.assertNotIn("segmentation",str(observation))
        self.assertNotIn("target_world",str(observation))
        self.assertNotIn("return_target",str(observation))
        observation["camera"]["pose_local_ned_m"][0]=900
        self.assertEqual(frame,before)

    def test_scene_acceptance_retains_target_hidden_requirement(self):
        truth={"target":{"visible_pixel_count":0},"distractor":{"visible_pixel_count":500},"occluder":{"visible_pixel_count":1000}}
        self.assertEqual(scene_acceptance("target_hidden_with_distractor_at_old_target",truth)["status"],"PASS")
        truth["target"]["visible_pixel_count"]=1
        self.assertEqual(scene_acceptance("target_hidden_with_distractor_at_old_target",truth)["status"],"FAIL")
        truth["target"]["visible_pixel_count"]=1000
        truth["occluder"]["visible_pixel_count"]=0
        self.assertEqual(scene_acceptance("return_target_right",truth)["status"],"PASS")
        truth["distractor"]["visible_pixel_count"]=499
        self.assertEqual(scene_acceptance("return_target_right",truth)["status"],"FAIL")

    def test_plain_conditions_are_identical_scale_and_no_cue(self):
        cfg=self.protocol("UNRESOLVABLE_IDENTICAL")
        self.assertEqual(cfg["target_texture"],cfg["distractor_texture"])
        self.assertEqual(cfg["phases"][0]["target_world_ned_m"][1],-.8)
        self.assertEqual(cfg["phases"][2]["target_world_ned_m"][1],.8)
        self.assertEqual(cfg["phases"][2]["distractor_world_ned_m"][1],-.8)

    def test_hidden_distractor_on_old_camera_target_ray(self):
        cfg=self.protocol()
        camera=cfg["camera_world_ned_m"]
        target=cfg["phases"][0]["target_world_ned_m"]
        distractor=cfg["phases"][1]["distractor_world_ned_m"]
        self.assertAlmostEqual((distractor[1]-camera[1])/(distractor[0]-camera[0]),(target[1]-camera[1])/(target[0]-camera[0]))

    def test_waits_are_at_most_point_two_seconds(self):
        times=[0.0];sleeps=[]
        def sleep(duration):
            sleeps.append(duration);times[0]+=duration
        wait_until(1.0,clock=lambda:times[0],sleeper=sleep)
        self.assertGreaterEqual(times[0],1.0)
        self.assertTrue(all(0<duration<=.2 for duration in sleeps))


if __name__=="__main__":
    unittest.main()

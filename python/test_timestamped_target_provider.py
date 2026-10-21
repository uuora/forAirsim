"""Fake-RPC checks for ray geometry, timestamp domains, and rejection paths."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import airsim
import cv2
import numpy as np

from balloon_detector import detect_balloon
from timestamped_target_provider import TimestampedTargetProvider


class TickClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        result = self.value
        self.value += 0.01
        return result


class ScriptClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def make_responses(image=None, depth=None, quaternion=(0, 0, 0, 1),
                   position=(1, -2, -3), image_timestamp=1_000_000_000,
                   depth_timestamp=1_000_000_000):
    if image is None:
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 60), 20, (0, 0, 255), -1)
    if depth is None:
        depth = np.full(image.shape[:2], 4.0)
    success, encoded = cv2.imencode(".png", image)
    assert success
    q = SimpleNamespace(x_val=quaternion[0], y_val=quaternion[1], z_val=quaternion[2], w_val=quaternion[3])
    p = SimpleNamespace(x_val=position[0], y_val=position[1], z_val=position[2])
    rgb = SimpleNamespace(image_type=airsim.ImageType.Scene, image_data_uint8=encoded.tobytes(),
                          width=image.shape[1], height=image.shape[0], pixels_as_float=False, compress=True,
                          time_stamp=image_timestamp, camera_position=p, camera_orientation=q)
    range_image = SimpleNamespace(image_type=airsim.ImageType.DepthPerspective,
                                  image_data_float=depth.reshape(-1).tolist(),
                                  width=depth.shape[1], height=depth.shape[0], pixels_as_float=True, compress=False,
                                  time_stamp=depth_timestamp, camera_position=deepcopy(p), camera_orientation=deepcopy(q))
    return [rgb, range_image]


class FakeClient:
    def __init__(self, responses=None, state_timestamp=1_020_000_000):
        self.responses = responses if responses is not None else make_responses()
        self.state = SimpleNamespace(timestamp=state_timestamp)
        self.calls = []
        self.requests = None

    def simGetImages(self, requests, vehicle_name):
        self.calls.append(("images", vehicle_name))
        self.requests = requests
        return self.responses

    def getMultirotorState(self, vehicle_name):
        self.calls.append(("state", vehicle_name))
        return self.state


class TimestampedProviderTests(unittest.TestCase):
    def capture(self, client=None, clock=None, **options):
        client = client or FakeClient()
        provider = TimestampedTargetProvider(client, monotonic_clock=clock or TickClock(), **options)
        return provider.capture("DroneA")

    def test_one_pair_rpc_then_state_and_json_metadata(self):
        client = FakeClient()
        observation = self.capture(client)
        self.assertTrue(observation.valid, observation.rejection_reasons)
        self.assertEqual(client.calls, [("images", "DroneA"), ("state", "DroneA")])
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0].image_type, airsim.ImageType.Scene)
        self.assertFalse(client.requests[0].pixels_as_float)
        self.assertTrue(client.requests[0].compress)
        self.assertEqual(client.requests[1].image_type, airsim.ImageType.DepthPerspective)
        self.assertTrue(client.requests[1].pixels_as_float)
        self.assertFalse(client.requests[1].compress)
        metadata = observation.metadata()
        json.dumps(metadata, allow_nan=False)
        self.assertNotIn("image", metadata)
        self.assertNotIn("depth", metadata)
        self.assertEqual(observation.image.shape, (160, 200, 3))
        self.assertEqual(observation.depth.shape, (160, 200))
        self.assertIs(observation.state_sample, client.state)
        self.assertEqual(metadata["rgb_render_readback_timestamp_ns"], 1_000_000_000)
        self.assertEqual(metadata["query_state_timestamp_ns"], 1_020_000_000)
        metadata["status"] = "tampered"
        self.assertEqual(observation.status, "OK")

    def test_ray_range_is_not_planar_depth_and_world_translation_is_explicit(self):
        observation = self.capture()
        metadata = observation.metadata()
        local = np.array(observation.estimate_local_ned())
        displacement = local - np.array(observation.camera_position_local_ned_m)
        self.assertAlmostEqual(np.linalg.norm(displacement), 4.0, places=12)
        self.assertLess(displacement[0], 4.0)  # Target is above principal pixel, so range != forward-axis depth.
        self.assertLess(displacement[2], 0.0)
        self.assertEqual(metadata["point_semantics"], "visible_surface_representative_point_not_balloon_center")
        with self.assertRaises(ValueError):
            observation.estimate_world()
        for translation in ([1, 2], [1, float("nan"), 3]):
            with self.assertRaises(ValueError):
                observation.estimate_world(translation)
        np.testing.assert_allclose(observation.estimate_world([10, 20, 30]), local + [10, 20, 30])

    def test_quaternion_rotates_ray_into_local_ned(self):
        q = (0, 0, np.sqrt(.5), np.sqrt(.5))
        observation = self.capture(FakeClient(make_responses(quaternion=q)))
        camera_point = np.array(observation.metadata()["projection"]["point_camera_m"])
        expected_rotated = np.array([-camera_point[1], camera_point[0], camera_point[2]])
        np.testing.assert_allclose(np.array(observation.estimate_local_ned())
                                   - np.array(observation.camera_position_local_ned_m), expected_rotated, atol=1e-12)

    def test_age_keeps_clock_components_and_does_not_use_rpc_duration(self):
        # A long image RPC does not become the readback age.
        clock = ScriptClock([10.0, 10.8, 10.8, 10.9, 10.91, 10.93])
        observation = self.capture(clock=clock)
        self.assertTrue(observation.valid)
        metadata = observation.metadata()
        self.assertAlmostEqual(metadata["timing"]["image_rpc_duration_s"], .8)
        self.assertAlmostEqual(observation.capture_age_at_receipt_s(), .02)
        self.assertAlmostEqual(observation.age_at(11.0), .12)
        components = observation.metadata()["age_components_last_evaluation"]
        self.assertAlmostEqual(components["image_readback_age_sim_s"], .02)
        self.assertAlmostEqual(components["host_processing_delay_s"], .1)
        self.assertFalse(metadata["clock_domains"]["host_epoch_used"])
        self.assertIn("not_capture_to_execute_latency", metadata["clock_domains"]["scope"])
        with self.assertRaises(ValueError):
            observation.age_at(10.89)
        with self.assertRaises(ValueError):
            observation.capture_age_at_receipt_s(1_030_000_000)  # Other query needs its host receipt anchor.
        self.assertAlmostEqual(observation.capture_age_at_receipt_s(1_030_000_000, 11.0), .03)
        self.assertAlmostEqual(observation.age_at(11.04), .07)

    def test_inference_completion_stale_proxy_rejects_estimate(self):
        clock = ScriptClock([10, 10.1, 10.1, 10.2, 10.21, 10.9])
        observation = self.capture(clock=clock)
        self.assertEqual(observation.status, "STALE")
        self.assertIn("OBSERVATION_STALE_AT_INFERENCE_COMPLETION", observation.rejection_reasons)
        with self.assertRaises(ValueError):
            observation.estimate_local_ned()

    def test_future_and_noninteger_timestamps_reject(self):
        observations = [self.capture(FakeClient(state_timestamp=999_999_999)),
                        self.capture(FakeClient(make_responses(image_timestamp=0))),
                        self.capture(FakeClient(make_responses(depth_timestamp=1.0e9))),
                        self.capture(FakeClient(state_timestamp=True))]
        for observation in observations:
            self.assertFalse(observation.valid)
            self.assertFalse(observation.metadata()["clock_valid"])
            with self.assertRaises(ValueError):
                observation.age_at(20)

    def test_reset_requires_explicit_acknowledgement(self):
        client = FakeClient()
        provider = TimestampedTargetProvider(client, monotonic_clock=TickClock())
        self.assertTrue(provider.capture("DroneA").valid)
        client.responses = make_responses(image_timestamp=500_000_000, depth_timestamp=500_000_000)
        client.state.timestamp = 520_000_000
        rejected = provider.capture("DroneA")
        self.assertIn("SIM_CLOCK_RESET_OR_REGRESSION_REQUIRES_EXPLICIT_ACKNOWLEDGEMENT", rejected.rejection_reasons)
        provider.reset_clock_tracking("DroneA")
        self.assertTrue(provider.capture("DroneA").valid)

    def test_rgb_depth_timestamp_skew_is_rejected_and_recorded(self):
        observation = self.capture(FakeClient(make_responses(depth_timestamp=1_030_000_000),
                                              state_timestamp=1_040_000_000))
        self.assertFalse(observation.valid)
        self.assertIn("RGB_DEPTH_TIMESTAMP_SKEW_EXCEEDED", observation.rejection_reasons)
        self.assertAlmostEqual(observation.metadata()["rgb_depth_skew_s"], .03)

    def test_camera_quaternion_and_paired_pose_consistency(self):
        for q in ((0, 0, 0, 0), (0, 0, 0, 2), (0, float("nan"), 0, 1)):
            with self.subTest(q=q):
                self.assertFalse(self.capture(FakeClient(make_responses(quaternion=q))).valid)
        responses = make_responses()
        responses[1].camera_position.x_val += .1
        observation = self.capture(FakeClient(responses))
        self.assertIn("RGB_DEPTH_CAMERA_POSE_SKEW_EXCEEDED", observation.rejection_reasons)
        self.assertAlmostEqual(observation.metadata()["rgb_depth_pose_translation_m"], .1)

    def test_no_detection_bad_rgb_and_bad_depth_reject(self):
        observations = []
        no_target = np.zeros((160, 200, 3), np.uint8)
        observations.append(self.capture(FakeClient(make_responses(image=no_target))))
        bad_png = make_responses()
        bad_png[0].image_data_uint8 = b"bad_png"
        observations.append(self.capture(FakeClient(bad_png)))
        bad_depth = make_responses()
        bad_depth[1].image_data_float = [1.0]
        observations.append(self.capture(FakeClient(bad_depth)))
        no_valid_depth = make_responses(depth=np.full((160, 200), np.nan))
        observations.append(self.capture(FakeClient(no_valid_depth)))
        self.assertEqual(observations[0].status, "NO_DETECTION")
        self.assertEqual(observations[-1].status, "NO_VALID_DEPTH")
        for observation in observations:
            self.assertFalse(observation.valid)
            json.dumps(observation.metadata(), allow_nan=False)
            with self.assertRaises(ValueError):
                observation.estimate_world([0, 0, 0])

    def test_resolution_difference_is_not_assumed_registered(self):
        observation = self.capture(FakeClient(make_responses(depth=np.full((80, 100), 4.0))))
        self.assertIn("RGB_DEPTH_RESOLUTION_MISMATCH_REQUIRES_AUDITED_NORMALIZED_INTRINSICS",
                      observation.rejection_reasons)

    def test_low_inner_box_depth_valid_fraction_rejects(self):
        depth = np.full((160, 200), np.nan)
        depth[60, 100] = 4.0
        observation = self.capture(FakeClient(make_responses(depth=depth)))
        self.assertEqual(observation.status, "NO_VALID_DEPTH")
        stats = observation.metadata()["depth_statistics"]
        self.assertEqual(stats["valid_count"], 1)
        self.assertLess(stats["valid_fraction"], .5)

    def test_representative_point_uses_actual_pixel_ray_and_range(self):
        responses = make_responses()
        decoded = cv2.imdecode(np.frombuffer(responses[0].image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        x, y, w, h = detect_balloon(decoded)["selected"]["bbox_xywh"]
        x0, x1 = int(np.floor(x+.25*w)), int(np.ceil(x+.75*w))
        y0, y1 = int(np.floor(y+.25*h)), int(np.ceil(y+.75*h))
        depth = np.full((160, 200), 4.0)
        region = np.full((y1-y0, x1-x0), 2.0)
        usable_count = region.size if region.size % 2 == 0 else region.size - 1
        region.reshape(-1)[usable_count // 2:usable_count] = 8.0
        if usable_count != region.size:
            region.reshape(-1)[-1] = np.nan
        depth[y0:y1, x0:x1] = region
        observation = self.capture(FakeClient(make_responses(depth=depth)))
        self.assertTrue(observation.valid, observation.rejection_reasons)
        metadata = observation.metadata()
        self.assertAlmostEqual(metadata["depth_statistics"]["median_range_m"], 5.0)
        self.assertIn(observation.depth_m, (2.0, 8.0))
        u, v = metadata["depth_statistics"]["representative_pixel_xy"]
        self.assertEqual(observation.depth[v, u], observation.depth_m)
        self.assertAlmostEqual(np.linalg.norm(metadata["projection"]["point_camera_m"]), observation.depth_m)

    def test_unconfirmed_frame_never_yields_world_position(self):
        observation = self.capture(coordinate_frame="UNCONFIRMED")
        self.assertIn("CAMERA_COORDINATE_FRAME_UNCONFIRMED", observation.rejection_reasons)
        with self.assertRaises(ValueError):
            observation.estimate_world([0, 0, 0])

    def test_bad_response_count_and_invalid_host_clock_return_json_rejections(self):
        for responses in ([], [make_responses()[0]], object()):
            observation = self.capture(FakeClient(responses))
            self.assertIn("PAIRED_IMAGE_RESPONSE_COUNT_INVALID", observation.rejection_reasons)
            json.dumps(observation.metadata(), allow_nan=False)
        observation = self.capture(clock=ScriptClock([10, float("nan"), 10.2, 10.3]))
        self.assertIn("HOST_MONOTONIC_CLOCK_REGRESSION_OR_INVALID", observation.rejection_reasons)
        json.dumps(observation.metadata(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import airsim
import cv2
import numpy as np
from t1_capture_diagnostics_v1 import capture,CaptureRejected


def response():
    ok,encoded=cv2.imencode('.png',np.full((3,4,3),128,np.uint8));assert ok
    return SimpleNamespace(width=4,height=3,image_data_uint8=encoded.tobytes(),image_data_float=[],
        camera_position=airsim.Vector3r(),camera_orientation=airsim.Quaternionr(),time_stamp=1000000000)


class Client:
    def __init__(self):
        self.responses=[response(),response(),response()]
        self.responses[1].image_data_uint8=b'';self.responses[1].image_data_float=[3.]*12
        self.calls=0;self.fail=False
    def simGetImages(self,*args,**kwargs):
        self.calls+=1
        if self.fail:raise RuntimeError('private service response must not be logged')
        return self.responses


class CaptureTests(unittest.TestCase):
    def setUp(self):self.client=Client();self.events=[]
    def run_capture(self):return capture(self.client,log=self.events.append,frame_id='frame-1')
    def test_success_keeps_rpc_receipt_before_decode_completion(self):
        image,depth,seg,meta=self.run_capture()
        self.assertEqual(image.shape,(3,4,3));self.assertEqual(depth.shape,(3,4))
        done=self.events[-1];self.assertEqual(done['kind'],'CAPTURE_PAIRED_COMPLETE')
        self.assertLessEqual(meta['capture_received_host_s'],done['recorded_monotonic_s'])
        self.assertFalse(done['source_clock_rebased']);self.assertEqual(self.client.calls,1)
        self.assertNotIn('image_data_uint8',str(self.events))
    def test_empty_rgb_has_exact_stage_and_never_calls_decoder_or_retries(self):
        self.client.responses[0].image_data_uint8=b''
        with patch('t1_capture_diagnostics_v1.cv2.imdecode') as decode:
            with self.assertRaisesRegex(CaptureRejected,'RGB_EMPTY'):self.run_capture()
            decode.assert_not_called()
        self.assertEqual(self.events[-1]['capture_stage'],'RGB_DECODE');self.assertEqual(self.client.calls,1)
        self.assertEqual(self.events[-1]['responses'][0]['png_byte_count'],0)
    def test_empty_segmentation_is_separate_from_rgb_failure(self):
        self.client.responses[2].image_data_uint8=b''
        with self.assertRaisesRegex(CaptureRejected,'SEGMENTATION_EMPTY'):self.run_capture()
        self.assertEqual(self.events[-1]['capture_stage'],'SEGMENTATION_DECODE')
    def test_bad_png_returns_explicit_decode_rejection(self):
        self.client.responses[0].image_data_uint8=b'invalid PNG'
        with self.assertRaisesRegex(CaptureRejected,'DECODE_RETURNED_NONE'):self.run_capture()
    def test_decoder_error_records_module_and_type_without_body(self):
        with patch('t1_capture_diagnostics_v1.cv2.imdecode',side_effect=cv2.error('private raw decoder body')):
            with self.assertRaises(cv2.error):self.run_capture()
        self.assertEqual(self.events[-1]['exception_module'],'cv2')
        self.assertEqual(self.events[-1]['exception_type'],'error')
        self.assertNotIn('private',str(self.events))
    def test_depth_count_is_checked_before_reshape(self):
        self.client.responses[1].image_data_float=[3.]*11
        with self.assertRaisesRegex(CaptureRejected,'DEPTH_LENGTH'):self.run_capture()
        self.assertEqual(self.events[-1]['capture_stage'],'DEPTH_RESHAPE')
    def test_header_dimensions_cannot_silently_disagree_with_png(self):
        self.client.responses[0].width=5
        with self.assertRaisesRegex(CaptureRejected,'SHAPES_MISMATCH'):self.run_capture()
    def test_response_count_retains_single_request_no_retry(self):
        self.client.responses.pop()
        with self.assertRaisesRegex(CaptureRejected,'THREE_PAIRED'):self.run_capture()
        self.assertEqual(self.client.calls,1)
    def test_time_skew_limit_stays_003_seconds(self):
        self.client.responses[2].time_stamp+=31000000
        with self.assertRaisesRegex(CaptureRejected,'TIMESTAMP_PAIRING'):self.run_capture()
    def test_pose_pairing_limit_stays_one_centimeter(self):
        self.client.responses[2].camera_position.x_val=.011
        with self.assertRaisesRegex(CaptureRejected,'POSE_PAIRING'):self.run_capture()
    def test_rpc_failure_is_categorical_and_contains_no_response_body(self):
        self.client.fail=True
        with self.assertRaises(RuntimeError):self.run_capture()
        self.assertEqual(self.events[-1]['capture_stage'],'RPC_IMAGES')
        self.assertNotIn('private',str(self.events))
    def test_zero_or_invalid_response_dimensions_fail_before_decode(self):
        self.client.responses[0].width=0
        with self.assertRaisesRegex(CaptureRejected,'DIMENSIONS'):self.run_capture()


if __name__=='__main__':unittest.main()

import unittest
import airsim
from test_t1_capture_diagnostics_v1 import Client
from t1_capture_diagnostics_v1 import CaptureRejected
from t1_capture_diagnostics_v2 import capture


class OrderedClient(Client):
    def simGetImages(self,requests,**kwargs):
        self.calls+=1;self.order=[r.image_type for r in requests]
        mapping={0:self.responses[0],2:self.responses[1],5:self.responses[2]}
        for kind,response in mapping.items():response.image_type=kind
        return [mapping[r.image_type] for r in requests]


class DepthFirstTests(unittest.TestCase):
    def setUp(self):self.client=OrderedClient();self.events=[]
    def run_capture(self):return capture(self.client,log=self.events.append,frame_id='frame-1')
    def test_request_is_single_depth_first_and_logical_return_remains_rgb_depth_seg(self):
        image,depth,seg,meta=self.run_capture()
        self.assertEqual(self.client.order,[2,0,5]);self.assertEqual(self.client.calls,1)
        self.assertEqual(image.shape,(3,4,3));self.assertEqual(depth.shape,(3,4))
        self.assertEqual(meta['response_camera_poses'][0]['render_readback_timestamp_ns'],1000000000)
    def test_skew_rejection_still_thirty_ms_and_no_retry(self):
        self.client.responses[2].time_stamp+=31000000
        with self.assertRaisesRegex(CaptureRejected,'TIMESTAMP_PAIRING'):self.run_capture()
        self.assertEqual(self.client.calls,1)
    def test_camera_pose_limit_unchanged(self):
        self.client.responses[2].camera_position.x_val=.011
        with self.assertRaisesRegex(CaptureRejected,'POSE_PAIRING'):self.run_capture()
    def test_zero_timestamp_cannot_be_rebased(self):
        self.client.responses[1].time_stamp=0
        with self.assertRaisesRegex(CaptureRejected,'TIMESTAMP_PAIRING'):self.run_capture()
    def test_response_type_binding_required(self):
        class Wrong(OrderedClient):
            def simGetImages(self,*args,**kwargs):
                result=super().simGetImages(*args,**kwargs);result[0].image_type=0;return result
        self.client=Wrong()
        with self.assertRaisesRegex(ValueError,'RESPONSE_TYPE'):self.run_capture()

if __name__=='__main__':unittest.main()

import time
import unittest
from t1_identity_flight import Flight,planned_slots,frame_from_dict,save
import tempfile
from pathlib import Path
import json
import numpy as np
from test_t1_temporal_identity import frame


class FakeClient:
    def __init__(self):self.commands=[]
    def moveByVelocityZAsync(self,*args,**kwargs):self.commands.append((args,kwargs))


class FlightDispatchTests(unittest.TestCase):
    def setUp(self):
        self.client=FakeClient();self.logs=[]
        self.flight=Flight(self.client,[0.,-3.,-1.],-3.,[-.3,-3.3,-3.15],[.3,-2.7,-2.85],self.logs.append)

    def test_expired_fresh_self_state_refuses_dispatch(self):
        gate={'latest_dispatch_monotonic_s':time.monotonic()-.01,
              'command_end_deadline_monotonic_s':time.monotonic()+.5,'current_binding':{}}
        with self.assertRaises(RuntimeError):self.flight.issue([.01,0.],.12,'CURRENT_TARGET_TRAJECTORY',gate=gate)
        self.assertEqual(self.client.commands,[])

    def test_end_deadline_checks_full_lease(self):
        now=time.monotonic()
        gate={'latest_dispatch_monotonic_s':now+.5,'command_end_deadline_monotonic_s':now+.01,'current_binding':{}}
        with self.assertRaises(RuntimeError):self.flight.issue([.01,0.],.12,'CURRENT_TARGET_TRAJECTORY',gate=gate)
        self.assertEqual(self.client.commands,[])

    def test_bound_current_command_uses_local_height(self):
        now=time.monotonic();gate={'latest_dispatch_monotonic_s':now+.12,
            'command_end_deadline_monotonic_s':now+.4,'current_binding':{'frame_id':'f1'}}
        row=self.flight.issue([.01,0.],.12,'CURRENT_TARGET_TRAJECTORY',gate=gate)
        self.assertEqual(self.client.commands[0][0][2],-2.)
        self.assertEqual(row['target_frame_id'],'f1')
        self.assertGreater(self.flight.target_lease_end,now)

    def test_own_brake_never_labels_target_frame(self):
        now=time.monotonic();gate={'latest_dispatch_monotonic_s':now+.12,'current_binding':{'frame_id':'old'}}
        row=self.flight.issue([0.,0.],.12,'OWN_STATE_LOSS_BRAKE',gate=gate)
        self.assertIsNone(row['target_frame_id']);self.assertIsNone(self.flight.target_lease_end)

    def test_nominal_slots_include_clipped_phase_end(self):
        slots,total=planned_slots({'period_s':.4,'phases':[{'phase_id':'visible','duration_s':7.},
            {'phase_id':'hidden','duration_s':2.},{'phase_id':'return','duration_s':7.}]})
        self.assertEqual(len(slots),41);self.assertEqual(total,16.)
        self.assertEqual(slots[17]['deadline_offset_s'],7.)
        self.assertEqual(slots[18]['offset_s'],7.)

    def test_frame_roundtrip_preserves_binding_and_candidate(self):
        source=frame(1,10.)
        self.assertEqual(frame_from_dict(source.to_dict()).to_dict(),source.to_dict())

    def test_numpy_scalar_report_keeps_number_types_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';save(path,{'count':np.int32(3),'value':np.float32(.5)})
            self.assertEqual(json.loads(path.read_text()),{'count':3,'value':.5})
            with self.assertRaises(ValueError):save(path,{'value':np.float32(np.nan)})

if __name__=='__main__':unittest.main()

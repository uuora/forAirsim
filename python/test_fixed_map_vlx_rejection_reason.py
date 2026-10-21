"""Rejection diagnosis must distinguish transport failure from source attributes."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fixed_map_perception import PerceptionService
class RejectionReasonTests(unittest.TestCase):
    def context(self):
        service=PerceptionService.__new__(PerceptionService);service.output=Path('unused');service.cfg={'source_attribute_guard':True};service.events=[]
        service.event=lambda kind,**data:service.events.append(dict(kind=kind,**data))
        request=SimpleNamespace(binding=lambda:{'request_id':'synthetic'},description='white-band red balloon',frame=SimpleNamespace(frame_id='synthetic-frame'))
        reply=SimpleNamespace(binding={'request_id':'synthetic'},completed_monotonic_s=1.,result={});service.backend=lambda request:reply
        return service,request
    def test_attribute_rejection_is_not_reported_as_backend_failure(self):
        s,r=self.context()
        with patch('t1_source_attribute_guard.verify_source_attribute',return_value={'status':'REJECTED','reason':'DESCRIBED_BAND_NOT_VERIFIED'}):
            with self.assertRaisesRegex(RuntimeError,'SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED_DESCRIBED_BAND_NOT_VERIFIED'):s._backend(r)
        self.assertEqual(s.events[-1]['reason'],'SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED_DESCRIBED_BAND_NOT_VERIFIED')
    def test_backend_failure_still_rejected(self):
        s,r=self.context();s.backend=lambda request:(_ for _ in ()).throw(ValueError('synthetic transport error'))
        with self.assertRaisesRegex(RuntimeError,'VLX_BACKEND_FAILURE_ValueError'):s._backend(r)
        self.assertEqual(s.events[-1]['reason'],'BACKEND_ValueError')
if __name__=='__main__':unittest.main()

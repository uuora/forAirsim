"""RGB descriptor prerequisites with artificial pixels, no API or truth."""
import hashlib
from dataclasses import replace
import unittest
import cv2
import numpy as np
from t1_source_attribute_guard import band_evidence,verify_source_attribute,AttributeGroundingBackend,SourceAttributeRejection
from t1_semantic_perception import Candidate,SemanticFrame,SemanticRequest,SemanticReply


def request_reply(image,*,description='带白色横条的红色气球'):
    ok,encoded=cv2.imencode('.png',image);assert ok
    data=encoded.tobytes();sha=hashlib.sha256(data).hexdigest()
    frame=SemanticFrame('s',1,'f0',0,sha,100,100,10.,10.01,
        (Candidate('target',(10.,10.,90.,90.),.8,'balloon'),),data)
    req=SemanticRequest('r0','s',1,description,hashlib.sha256(description.encode()).hexdigest(),10.02,frame)
    reply=SemanticReply(req.binding(),10.5,{'result_bbox_list':[dict(xmin=10.,ymin=10.,xmax=90.,ymax=90.)]})
    return req,reply


class AttributeTests(unittest.TestCase):
    def setUp(self):
        self.plain=np.full((100,100,3),(30,30,220),np.uint8)
        self.band=self.plain.copy();self.band[40:60,10:90]=230
    def test_broad_horizontal_band_with_red_above_below(self):
        self.assertEqual(band_evidence(self.band,[10.,10.,90.,90.])['status'],'ACCEPTED')
    def test_plain_red_and_narrow_specular_highlight_rejected(self):
        glare=self.plain.copy();cv2.circle(glare,(50,30),7,(220,220,240),-1)
        for image in (self.plain,glare):
            self.assertEqual(band_evidence(image,[10.,10.,90.,90.])['status'],'REJECTED')
    def test_white_uniform_and_vertical_strip_rejected(self):
        vertical=self.plain.copy();vertical[:,45:55]=230
        for image in (np.full((100,100,3),230,np.uint8),vertical):
            self.assertEqual(band_evidence(image,[10.,10.,90.,90.])['status'],'REJECTED')
    def test_thin_horizontal_line_and_white_top_cap_rejected(self):
        thin=self.plain.copy();thin[48:51]=230
        top=self.plain.copy();top[:45]=230
        for image in (thin,top):
            self.assertEqual(band_evidence(image,[10.,10.,90.,90.])['status'],'REJECTED')
    def test_small_truncated_or_invalid_box_rejected(self):
        for box in ([1.,1.,15.,15.],[-1.,10.,90.,90.],[10.,10.,float('nan'),90.],[True,10.,90.,90.]):
            self.assertEqual(band_evidence(self.band,box)['status'],'REJECTED')
    def test_bound_source_and_unmodified_reply(self):
        req,reply=request_reply(self.band)
        audit=verify_source_attribute(req,reply)
        self.assertEqual(audit['status'],'ACCEPTED')
        self.assertEqual(audit['request_binding'],req.binding())
        self.assertEqual(audit['source_candidate_id'],'target')
        self.assertFalse(audit['flight_authorized'])
    def test_unsupported_description_and_corrupt_bytes_rejected(self):
        req,reply=request_reply(self.band,description='another red object')
        self.assertEqual(verify_source_attribute(req,reply)['reason'],'UNSUPPORTED_DESCRIPTOR_PROFILE')
        req,reply=request_reply(self.band)
        object.__setattr__(req.frame,'encoded_image',b'changed')
        self.assertEqual(verify_source_attribute(req,reply)['reason'],'SOURCE_RGB_BYTES_BINDING_MISMATCH')
    def test_backend_rejects_false_source_without_rewriting_reply(self):
        req,reply=request_reply(self.plain)
        class Backend:
            vlx_client=None
            active_record={}
            def __call__(self,request):return reply
            def mark(self,record,**kw):record.update(kw)
        backend=Backend();guard=AttributeGroundingBackend(backend)
        with self.assertRaises(SourceAttributeRejection):guard(req)
        self.assertEqual(backend.active_record['source_attribute_guard_status'],'REJECTED')
        self.assertEqual(reply.result['result_bbox_list'][0]['xmin'],10.)
    def test_out_of_bounds_semantic_box_is_not_attribute_accepted(self):
        req,reply=request_reply(self.band)
        reply.result['result_bbox_list'][0]['xmin']=-1.
        self.assertEqual(verify_source_attribute(req,reply)['reason'],'SEMANTIC_SOURCE_BOX_OUT_OF_BOUNDS_OR_UNORDERED')
    def test_truncated_source_candidate_is_rejected(self):
        req,reply=request_reply(self.band)
        object.__setattr__(req.frame,'candidates',(replace(req.frame.candidates[0],truncated=True),))
        self.assertEqual(verify_source_attribute(req,reply)['reason'],'SOURCE_CANDIDATE_TRUNCATED')


if __name__=='__main__':unittest.main()

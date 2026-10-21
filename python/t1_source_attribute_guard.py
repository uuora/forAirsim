"""Development RGB guard for the declared red/white-band balloon descriptor.

This is a conservative source-frame prerequisite, not identity probability or
appearance continuity. It uses only the bound RGB and returned/source boxes.
Thresholds are fixed here before prospective validation; lighting, small boxes,
other attributes and strong viewpoint change are outside this initial scope.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
import cv2
import numpy as np
from t1_semantic_perception import box_iou


@dataclass(frozen=True)
class AttributeConfig:
    minimum_box_size_px:int=24
    white_minimum_channel:float=105.
    white_maximum_relative_chroma:float=.20
    band_row_white_fraction:float=.65
    minimum_band_height_fraction:float=.12
    minimum_red_fraction:float=.15
    minimum_red_row_fraction:float=.45


class SourceAttributeRejection(RuntimeError):
    pass


def band_evidence(bgr, bbox, config=AttributeConfig()):
    """Require one broad white horizontal band with red above and below."""
    result=dict(status='REJECTED',reason='INVALID_RGB_OR_BOX',configuration=asdict(config),
        scope='source RGB descriptor evidence only; not calibrated instance identity',flight_authorized=False)
    if not isinstance(bgr,np.ndarray) or bgr.ndim!=3 or bgr.shape[2]!=3 or bgr.dtype!=np.uint8:return result
    if not isinstance(bbox,(list,tuple)) or len(bbox)!=4 or not all(
            isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in bbox):return result
    x0,y0,x1,y1=bbox;h,w=bgr.shape[:2]
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:return result
    bw,bh=x1-x0,y1-y0
    if min(bw,bh)<config.minimum_box_size_px:return dict(result,reason='SOURCE_ROI_TOO_SMALL')
    # Inner rectangle reduces sky/background and rounded silhouette influence.
    xa,xb=int(math.ceil(x0+.25*bw)),int(math.floor(x1-.25*bw))
    ya,yb=int(math.ceil(y0+.10*bh)),int(math.floor(y1-.10*bh))
    core=bgr[ya:yb,xa:xb].astype(float)
    low,high=core.min(axis=2),core.max(axis=2)
    white=(low>=config.white_minimum_channel)&((high-low)<=config.white_maximum_relative_chroma*np.maximum(high,1.))
    blue,green,red=core[:,:,0],core[:,:,1],core[:,:,2]
    red_mask=(red>=70.)&(red>=1.4*np.maximum(blue,green))
    white_rows=white.mean(axis=1);red_rows=red_mask.mean(axis=1)
    runs=[];start=None
    for index,accepted in enumerate(list(white_rows>=config.band_row_white_fraction)+[False]):
        if accepted and start is None:start=index
        if not accepted and start is not None:runs.append((start,index));start=None
    required=max(3,int(math.ceil(config.minimum_band_height_fraction*bh)))
    valid=[(start,end) for start,end in runs if end-start>=required and
        np.count_nonzero(red_rows[:start]>=config.minimum_red_row_fraction)>=2 and
        np.count_nonzero(red_rows[end:]>=config.minimum_red_row_fraction)>=2]
    red_fraction=float(red_mask.mean())
    accepted=bool(valid and red_fraction>=config.minimum_red_fraction)
    return dict(result,status='ACCEPTED' if accepted else 'REJECTED',
        reason='RED_WITH_BROAD_WHITE_HORIZONTAL_BAND' if accepted else 'DESCRIBED_BAND_NOT_VERIFIED',
        core_roi_xyxy=[xa,ya,xb,yb],core_red_fraction=red_fraction,
        broad_white_band_count=len(valid),minimum_band_rows=required,
        longest_white_row_run=max((end-start for start,end in runs),default=0),
        white_fraction_per_row=white_rows.tolist(),red_fraction_per_row=red_rows.tolist())


def verify_source_attribute(request,reply):
    frame=request.frame
    audit=dict(status='REJECTED',reason='SOURCE_ATTRIBUTE_INPUT_INVALID',
        request_binding=request.binding(),source_frame_id=frame.frame_id,source_image_sha256=frame.image_sha256,
        profile='RED_BALLOON_WITH_BROAD_WHITE_HORIZONTAL_BAND',flight_authorized=False)
    if reply.binding!=request.binding():return dict(audit,reason='REPLY_REQUEST_BINDING_MISMATCH')
    if request.description!='带白色横条的红色气球':return dict(audit,reason='UNSUPPORTED_DESCRIPTOR_PROFILE')
    if not frame.encoded_image or hashlib.sha256(frame.encoded_image).hexdigest()!=frame.image_sha256:
        return dict(audit,reason='SOURCE_RGB_BYTES_BINDING_MISMATCH')
    boxes=reply.result.get('result_bbox_list') if isinstance(reply.result,dict) else None
    if not isinstance(boxes,list) or len(boxes)!=1:return dict(audit,reason='NO_SINGLE_SEMANTIC_SOURCE_BOX')
    try:box=[boxes[0][key] for key in ('xmin','ymin','xmax','ymax')]
    except (KeyError,TypeError):return audit
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box):return audit
    if not 0<=box[0]<box[2]<=frame.width or not 0<=box[1]<box[3]<=frame.height:
        return dict(audit,reason='SEMANTIC_SOURCE_BOX_OUT_OF_BOUNDS_OR_UNORDERED')
    matches=sorted([(box_iou(box,c.bbox_xyxy_pixels),c) for c in frame.candidates],key=lambda row:-row[0])
    if not matches or matches[0][0]<.5 or (len(matches)>1 and (matches[1][0]>=.5 or matches[0][0]-matches[1][0]<.15)):
        return dict(audit,reason='SOURCE_BOX_NOT_UNIQUE_YOLO_CANDIDATE')
    if matches[0][1].truncated is not False:return dict(audit,reason='SOURCE_CANDIDATE_TRUNCATED')
    rgb=cv2.imdecode(np.frombuffer(frame.encoded_image,np.uint8),cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2]!=(frame.height,frame.width):return dict(audit,reason='SOURCE_RGB_GEOMETRY_MISMATCH')
    evidence=band_evidence(rgb,list(matches[0][1].bbox_xyxy_pixels))
    return dict(audit,status=evidence['status'],reason=evidence['reason'],
        source_candidate_id=matches[0][1].candidate_id,source_candidate=matches[0][1].to_dict(),
        semantic_source_bbox_xyxy_pixels=box,attribute_evidence=evidence)


class AttributeGroundingBackend:
    def __init__(self,backend):
        self.backend=backend;self.vlx_client=backend.vlx_client
    def warmup(self,*args,**kwargs):return self.backend.warmup(*args,**kwargs)
    def detect_frame(self,*args,**kwargs):return self.backend.detect_frame(*args,**kwargs)
    def snapshot(self):return self.backend.snapshot()
    def __call__(self,request):
        reply=self.backend(request)
        audit=verify_source_attribute(request,reply)
        self.backend.mark(self.backend.active_record,source_attribute_audit=audit,
            source_attribute_guard_status=audit['status'])
        if audit['status']!='ACCEPTED':raise SourceAttributeRejection('SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED')
        return reply


class AttributeGuardedFlightObserver:
    def __init__(self,detector_config_path):
        from t1_delayed_semantic_observer import DelayedFlightObserver
        self.inner=DelayedFlightObserver(detector_config_path)
        self.inner.core.backend=AttributeGroundingBackend(self.inner.core.backend)
        self.close_receipt=None
    def begin(self,*args,**kwargs):
        result=self.inner.begin(*args,**kwargs)
        result['source_attribute_guard']='BOUND_RGB_RED_WHITE_BAND_PREREQUISITE'
        return result
    def process(self,observation):
        result=self.inner.process(observation)
        result['source_attribute_guard']='BOUND_RGB_RED_WHITE_BAND_PREREQUISITE'
        return result
    def close(self):
        self.inner.close();self.close_receipt=dict(self.inner.close_receipt,
            source_attribute_guard='BOUND_RGB_RED_WHITE_BAND_PREREQUISITE')

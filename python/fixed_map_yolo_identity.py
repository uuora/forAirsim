"""YOLO class/appearance association only. No semantic API or scene truth.

Local track numbers express image continuity, not physical instance identity.
Every accepted frame has its own appearance audit; no stale review is adopted.
"""
import hashlib,time
from pathlib import Path
from dataclasses import dataclass
from t1_semantic_perception import YoloVlxBackend
from fixed_map_perception_execution import current_attribute_audit

class YoloImageBackend(YoloVlxBackend):
    def __init__(self,detector):
        self.detector=detector;self.clock=time.monotonic;self._warmed=False
    def __call__(self,*args,**kwargs):
        raise RuntimeError('IMAGE_ONLY_BACKEND_HAS_NO_SEMANTIC_CLIENT')
    @staticmethod
    def _image(path,expected_sha256=None):
        from fixed_map_frame_image import decode_bound_rgb
        encoded=Path(path).read_bytes();digest=hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('saved image hash changed')
        return encoded,digest,decode_bound_rgb(encoded,digest)

@dataclass(frozen=True)
class AssociationConfig:
    confirmation_frames:int=3
    minimum_iou:float=.25
    maximum_gap_s:float=.5

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x*y
    return x*y/union if union>0 else 0.

def select(frame,description):
    audit=current_attribute_audit(frame,description,None)
    if description=='气球':
        matches=[c.candidate_id for c in frame.candidates if not c.truncated]
    else:matches=audit.get('matching_candidate_ids',[])
    # Existing guard stores rows rather than a public matching list.
    if description!='气球' and not matches:
        matches=[r['candidate_id'] for r in audit.get('candidates',[]) if r.get('matches_description')]
    if len(matches)!=1:
        return None,dict(status='REJECTED',reason='NO_MATCHING_YOLO_CANDIDATE' if not matches else 'DESCRIPTOR_AMBIGUOUS_MULTIPLE_CURRENT_CANDIDATES',frame_id=frame.frame_id,image_sha256=frame.image_sha256,matching_candidate_ids=matches,attribute_details=audit)
    selected=matches[0]
    if description=='气球':
        audit=dict(status='ACCEPTED',reason='SINGLE_UNTRUNCATED_BALLOON_CLASS_CANDIDATE',frame_id=frame.frame_id,image_sha256=frame.image_sha256,selected_candidate_id=selected)
    else:
        # The initial audit already hashes this immutable source and evaluates
        # every candidate. Bind its unique result without decoding it again.
        audit=dict(audit,status='ACCEPTED',reason='UNIQUE_CURRENT_RGB_ATTRIBUTE_MATCH',selected_candidate_id=selected)
    return selected,audit

class YoloAssociation:
    def __init__(self,session,generation,description,config=None):
        self.session=session;self.generation=generation;self.description=description
        self.config=config or AssociationConfig();self.previous=None;self.count=0;self.serial=0;self.track_id=None
    def update(self,frame,semantic_decision=None,now=None):
        now=time.monotonic() if now is None else now
        selected,audit=select(frame,self.description)
        c=next((c for c in frame.candidates if c.candidate_id==selected),None)
        valid=(frame.session_id==self.session and frame.target_generation==self.generation and audit['status']=='ACCEPTED' and 0<=now-frame.readback_receipt_monotonic_s<=self.config.maximum_gap_s)
        reason=audit['reason'] if not valid else 'CONTINUOUS_IMAGE_ASSOCIATION_PENDING'
        if not valid:
            self.count=0;self.previous=None;self.track_id=None
        else:
            continuous=self.previous is not None and frame.frame_sequence>self.previous[0] and 0<frame.readback_receipt_monotonic_s-self.previous[1]<=self.config.maximum_gap_s and iou(c.bbox_xyxy_pixels,self.previous[2])>=self.config.minimum_iou
            if continuous:self.count+=1
            else:self.count=1;self.serial+=1;self.track_id=f'yolo-local-{self.serial:04d}'
            self.previous=(frame.frame_sequence,frame.readback_receipt_monotonic_s,c.bbox_xyxy_pixels)
        state='TRACKING' if valid and self.count>=self.config.confirmation_frames else ('ACQUIRING' if valid else 'LOST')
        if state=='TRACKING':reason='YOLO_CURRENT_ATTRIBUTE_AND_CONTINUITY_VALID'
        anchor=dict(kind='yolo_current_rgb_attributes_v8',session_id=self.session,target_generation=self.generation,description_sha256=hashlib.sha256(self.description.encode()).hexdigest(),frame_id=frame.frame_id,image_sha256=frame.image_sha256,selected_candidate_id=selected,consecutive_frames=self.count,track_id=self.track_id,identity_scope='local_appearance_continuity_NOT_truth_instance')
        return dict(state=state,reason=reason,selected_candidate_id=selected if valid else None,track_id=self.track_id,semantic_anchor=anchor,semantic_review_deadline_monotonic_s=frame.readback_receipt_monotonic_s+self.config.maximum_gap_s,attribute_audit=audit,consecutive_frames=self.count)

class DisabledReview:
    def poll(self,*args,**kwargs):return dict(status='DISABLED_YOLO_ONLY',reason='NO_VLX_REQUEST_OR_REPLY')
    def request(self,*args,**kwargs):return dict(status='DISABLED_YOLO_ONLY',reason='NO_VLX_REQUEST_OR_REPLY')
    def close(self):pass

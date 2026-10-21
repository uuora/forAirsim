"""Same-frame scale fallback; appearance is checked again by the consumer."""
import copy
import time

from t1_source_attribute_guard import band_evidence


def box_iou(a,b):
    width=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    height=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=width*height
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union>0 else 0.


class ConditionalScaleBalloonDetector:
    def __init__(self,detector,description='带白色横条的红色气球'):
        if detector.imgsz!=640:raise ValueError('CONDITIONAL_PRIMARY_SCALE_MUST_BE_640')
        self.detector=detector;self.description=description
        self.base_imgsz=640 if description=='带白色横条的红色气球' else 1280
        self.primary_imgsz=self.base_imgsz
    def detect(self,image):
        started=time.monotonic();primary_size=self.primary_imgsz
        try:
            self.detector.imgsz=primary_size;primary=self.detector.detect(image)
        finally:self.detector.imgsz=640
        def matches(candidate):
            if self.description=='黄色气球':
                from fixed_map_target_profiles import yellow_evidence
                return yellow_evidence(image,candidate['bbox_xyxy_pixels'],candidate['truncated'])['matched']
            if self.description=='气球':return not candidate['truncated']
            if self.description in ['红色气球','普通红色气球']:
                band=band_evidence(image,candidate['bbox_xyxy_pixels'])
                return not candidate['truncated'] and band.get('core_red_fraction',0)>=.15 and (self.description=='红色气球' or band['status']!='ACCEPTED')
            return not candidate['truncated'] and band_evidence(image,candidate['bbox_xyxy_pixels'])['status']=='ACCEPTED'
        existing=primary['candidates']
        fallback=None
        fallback_size=(960 if self.description=='带白色横条的红色气球' else 1280) if primary_size==640 else 640
        # A 640 miss may hide a small target entirely. Try the same current
        # image once at 960; never manufacture or carry forward old boxes.
        if (not existing and primary_size==640) or (existing and not any(matches(c) for c in existing)):
            original_size=self.detector.imgsz
            try:
                self.detector.imgsz=fallback_size;fallback=self.detector.detect(image)
            finally:self.detector.imgsz=original_size
        result=copy.deepcopy(primary)
        if fallback is not None:
            merged=list(result['candidates'])
            for candidate in fallback['candidates']:
                overlap=next((i for i,c in enumerate(merged) if box_iou(c['bbox_xyxy_pixels'],candidate['bbox_xyxy_pixels'])>=.5),None)
                if overlap is None:merged.append(copy.deepcopy(candidate))
                else:
                    old=merged[overlap]
                    # Duplicate geometry is one proposal. Prefer an actually
                    # visible band, then detector score; no track/truth IDs.
                    if (matches(candidate),candidate['confidence'])>(matches(old),old['confidence']):merged[overlap]=copy.deepcopy(candidate)
            merged.sort(key=lambda c:c['confidence'],reverse=True)
            result['candidates']=merged;result['candidate_count']=len(merged)
            if any(matches(c) for c in fallback['candidates']):self.primary_imgsz=fallback_size
        elif not existing:self.primary_imgsz=self.base_imgsz
        result['conditional_scale_receipt']=dict(primary_imgsz=primary_size,fallback_imgsz=fallback_size,next_primary_imgsz=self.primary_imgsz,fallback_used=fallback is not None,elapsed_host_s=time.monotonic()-started,source='same RGB array; weights/confidence/IoU unchanged',trigger='640 has no candidates, or visible candidates lack the described band',scale_memory_scope='image resolution only; no old box, target coordinates or identity carried into new frames',limitation='additional inference costs time; all current evidence age checks still apply',identity_scope='candidate proposals only; consumer performs unique descriptor/continuity checks')
        return result

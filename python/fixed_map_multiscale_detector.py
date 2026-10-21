"""Two frozen YOLO passes; full-image coordinates and deterministic cross-scale NMS."""
import copy,time,threading
from t1_semantic_perception import box_iou

class MultiScaleBalloonDetector:
    def __init__(self,detector,scales=(1280,640)):
        if tuple(scales)!=(1280,640):raise ValueError('UNDECLARED_INFERENCE_SCALES')
        self.detector=detector;self.scales=tuple(scales);self.lock=threading.Lock()
    def detect(self,bgr):
        started=time.perf_counter();passes=[];pool=[]
        with self.lock:
            previous=self.detector.imgsz
            try:
                for scale in self.scales:
                    self.detector.imgsz=scale;report=self.detector.detect(bgr);passes.append(dict(imgsz=scale,report=copy.deepcopy(report)))
                    for c in report['candidates']:pool.append(dict(copy.deepcopy(c),inference_imgsz=scale))
            finally:self.detector.imgsz=previous
        pool.sort(key=lambda c:(-c['confidence'],-c['inference_imgsz'],tuple(c['bbox_xyxy_pixels'])))
        kept=[];suppressed=[]
        for c in pool:
            parent=next((p for p in kept if box_iou(c['bbox_xyxy_pixels'],p['bbox_xyxy_pixels'])>self.detector.nms_iou),None)
            if parent is None:kept.append(c)
            else:suppressed.append(dict(candidate=c,suppressed_by_bbox=parent['bbox_xyxy_pixels']))
        result=copy.deepcopy(passes[0]['report']);result.update(source='Frozen specialist YOLO full-image dual-scale v7',candidate_count=len(kept),candidates=kept,multiscale=dict(scales=list(self.scales),passes=passes,suppressed=suppressed,cross_scale_nms_iou=self.detector.nms_iou,coordinates='original full RGB pixel coordinates; no target-conditioned crops',wall_s=time.perf_counter()-started),designated_instance_identity_established=False)
        return result

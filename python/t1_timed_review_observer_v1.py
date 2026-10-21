"""Same RGB/identity decisions with nested component clocks; no new inputs."""
from functools import wraps
import time

from t1_review_handoff_observer_v1p1 import ReacquisitionAwareFlightObserver
from t1_source_attribute_guard import AttributeGroundingBackend,verify_source_attribute,SourceAttributeRejection


class TimedMethods:
    def __init__(self,clock=time.monotonic):self.clock=clock;self.active=None

    def wrap(self,owner,name,label):
        original=getattr(owner,name)
        @wraps(original)
        def timed(*args,**kwargs):
            if self.active is None:return original(*args,**kwargs)
            trace=self.active;started=self.clock();status='RETURNED';error_type=None
            try:return original(*args,**kwargs)
            except Exception as error:
                status='RAISED';error_type=type(error).__name__;raise
            finally:trace.append(dict(component=label,started_monotonic_s=started,
                completed_monotonic_s=self.clock(),status=status,error_type=error_type))
        setattr(owner,name,timed)


class TimedAttributeBackend(AttributeGroundingBackend):
    def __call__(self,request):
        reply=self.backend(request)
        started=time.monotonic()
        audit=verify_source_attribute(request,reply)
        completed=time.monotonic()
        self.backend.mark(self.backend.active_record,source_attribute_audit=audit,
            source_attribute_guard_status=audit['status'],source_attribute_guard_started_monotonic_s=started,
            source_attribute_guard_completed_monotonic_s=completed)
        if audit['status']!='ACCEPTED':raise SourceAttributeRejection('SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED')
        return reply


class TimedReviewFlightObserver(ReacquisitionAwareFlightObserver):
    def __init__(self,detector_config_path):
        super().__init__(detector_config_path)
        core=self.inner.core
        core.backend=TimedAttributeBackend(core.backend.backend)
        self.timing=TimedMethods()
        self.timing.wrap(core.backend,'detect_frame','SAVED_RGB_READ_DETECTION_AND_CANDIDATES')
        self.timing.wrap(core.backend.backend.backend.detector,'detect','YOLO_DETECT_CALL')

    def begin(self,*args,**kwargs):
        result=super().begin(*args,**kwargs)
        core=self.inner.core
        for owner,name,label in [(core.policy,'poll','SEMANTIC_POLL'),(core.policy,'request','SEMANTIC_REQUEST_SCHEDULING'),
                                 (core.tracker,'update','CURRENT_FRAME_ASSOCIATION')]:
            self.timing.wrap(owner,name,label)
        result['component_timing_scope']='nested intervals; detector interval is inside detection/candidate interval, not additive'
        return result

    def process(self,observation):
        spans=[];self.timing.active=spans
        try:result=super().process(observation)
        finally:self.timing.active=None
        result['main_thread_component_timings']=spans
        result['component_timing_scope']='nested intervals; async VLX and source-attribute guard are separate transport-trace records'
        return result

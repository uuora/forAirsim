"""Hold real semantic replies in one worker; never rewrite their source clock."""
from copy import deepcopy
import threading
import time

from t1_delayed_semantic_observer import transport_error_category
from t1_review_handoff_observer_v1p1 import ReacquisitionAwareFlightObserver


class ReplyDeliveryFaultBackend:
    def __init__(self, backend, declaration, *, clock=time.monotonic, sleep=time.sleep):
        if declaration['durations_s'] != ([3.,9.] if declaration['enabled'] else []):
            raise ValueError('FROZEN_REPLY_DELIVERY_HOLDS_REQUIRED')
        self.backend=backend;self.vlx_client=backend.vlx_client
        self.declaration=deepcopy(declaration);self.clock=clock;self.sleep=sleep
        self.lock=threading.Lock();self.epoch=None;self.assigned=0;self.records=[]

    def activate(self, epoch):
        with self.lock:
            if self.epoch is not None:raise ValueError('MEASUREMENT_EPOCH_ALREADY_SET')
            self.epoch=epoch

    def warmup(self, *args, **kwargs):return self.backend.warmup(*args, **kwargs)
    def detect_frame(self, *args, **kwargs):return self.backend.detect_frame(*args, **kwargs)

    def snapshot(self):
        original=self.backend.snapshot()
        with self.lock:delivery=deepcopy(self.records)
        by_request={r['request_binding']['request_id']:r for r in delivery}
        for r in original:
            key=r['request_binding']['request_id']
            if key in by_request:r['controlled_reply_delivery']=by_request[key]
        return original

    def __call__(self, request):
        with self.lock:
            eligible=(self.declaration['enabled'] and self.epoch is not None
                and request.frame.readback_receipt_monotonic_s>=self.epoch
                and self.assigned<len(self.declaration['durations_s']))
            hold=self.declaration['durations_s'][self.assigned] if eligible else 0.
            if eligible:self.assigned+=1
            record=dict(request_binding=request.binding(),declared_hold_s=hold,
                measurement_epoch_monotonic_s=self.epoch,eligible_original_source=bool(eligible),
                service_reply_clock_and_content_unchanged=True,natural_service_delay=False,
                delivery_status='REAL_BACKEND_RUNNING')
            self.records.append(record)
        try:
            reply=self.backend(request)
            returned=self.clock()
            with self.lock:
                record.update(real_backend_returned_monotonic_s=returned,
                    original_reply_completed_monotonic_s=reply.completed_monotonic_s,
                    original_reply_binding=deepcopy(reply.binding),
                    delivery_status='CONTROLLED_HOLD' if hold else 'NO_HOLD')
            if hold:
                start=self.clock()
                with self.lock:record['hold_started_monotonic_s']=start
                self.sleep(hold)
                with self.lock:record['hold_completed_monotonic_s']=self.clock()
            with self.lock:record.update(delivered_to_policy_monotonic_s=self.clock(),delivery_status='DELIVERED')
            # A held reply retains the real service clock. The future becomes
            # done later; that later time is independently visible above.
            return reply
        except Exception as error:
            with self.lock:record.update(delivery_status='REAL_BACKEND_FAILED',
                failure_completed_monotonic_s=self.clock(),error_category=transport_error_category(error))
            raise


class PerturbedFlightObserver(ReacquisitionAwareFlightObserver):
    def __init__(self, detector_config_path, declaration):
        super().__init__(detector_config_path)
        self.delivery=ReplyDeliveryFaultBackend(self.inner.core.backend,declaration)
        self.inner.core.backend=self.delivery

    def start_measurement(self, epoch):self.delivery.activate(epoch)

"""Reject unsafe retained-review evidence while allowing the next HTTP request."""
from copy import deepcopy
import unittest
import test_t1_review_handoff as fixtures
from t1_pending_review_audit_v1 import pending_review_audit


class PendingReviewAuditTests(unittest.TestCase):
    def fixture(self):
        h=fixtures.HandoffTests();h.setUp();rows=[];poll=h.policy.poll
        def traced_poll(frame,**kwargs):
            result=poll(frame,**kwargs)
            rows.append(dict(frame_id=frame.frame_id,observer_result=dict(semantic_frame=frame.to_dict(),
                semantic_poll=deepcopy(result),process_completed_host_monotonic_s=frame.readback_receipt_monotonic_s+.04)))
            return result
        h.policy.poll=traced_poll
        try:
            h.initial();h.advance(4,15);h.review(15)
            h.submit(h.frame(16));h.step(17);h.step(18)
            for row,selection in zip(rows,h.results):row['observer_result']['temporal_identity']=deepcopy(selection)
            return dict(configuration=dict(description=fixtures.fixtures.DESCRIPTION),acquisition_frames=[],frames=rows,
                observer_close=dict(request_terminal_trace=dict(terminal_request_trace=deepcopy(h.policy._terminal_trace))))
        finally:h.tearDown()

    def test_pending_confirmation_and_http_request_ids_can_differ(self):
        report=self.fixture();result=pending_review_audit(report)
        self.assertTrue(result['status'].startswith('PASS'))
        differing=[r for r in result['rows'] if r['transport_and_pending_review_ids_differ']]
        self.assertEqual(len(differing),1);self.assertEqual(differing[0]['frame_id'],'frame-17')

    def test_current_candidate_change_is_not_accepted(self):
        report=self.fixture();report['frames'][17]['observer_result']['temporal_identity']['selected_candidate']['bbox_xyxy_pixels'][0]+=10.
        self.assertTrue(pending_review_audit(report)['status'].startswith('FAIL'))

    def test_original_accepted_review_is_required(self):
        report=self.fixture();report['observer_close']['request_terminal_trace']['terminal_request_trace'][1]['semantic_reply_sha256']='0'*64
        self.assertTrue(pending_review_audit(report)['status'].startswith('FAIL'))

    def test_future_frame_cannot_supply_current_confirmation(self):
        report=self.fixture();report['frames'][17]['observer_result']['temporal_identity']['review_handoff']['confirmation_count']=3
        self.assertTrue(pending_review_audit(report)['status'].startswith('FAIL'))


if __name__=='__main__':unittest.main()

"""Delayed transport clocks and irreversible route failures; no real API."""
from dataclasses import replace
import unittest
import test_t1_candidate_surface_v3 as fixtures
from t1_delayed_semantic_bridge import DelayedSemanticPolicy, DelayedBridgeConfig
from t1_semantic_perception import SemanticReply, Candidate


class DelayedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.executor=fixtures.ManualExecutor()
        self.clock_value=13.
        self.policy=self.make_policy()
    def make_policy(self, **config):
        return DelayedSemanticPolicy(lambda req: None,session_id='surface-session',target_generation=2,
            description=fixtures.DESCRIPTION,executor=self.executor,clock=lambda:self.clock_value,
            config=DelayedBridgeConfig(**config))
    def tearDown(self):
        self.clock_value=100.
        self.policy.close()
    def source(self):
        source=fixtures.make_frame(0,10.)
        self.assertEqual(self.policy.request(source,now=10.02)['status'],'SUBMITTED')
        return source
    def reply(self, completed, *, boxes=None, binding=None):
        future,req=self.executor.calls[-1]
        future.set_result(SemanticReply(binding or req.binding(),completed,{'result_bbox_list':boxes if boxes is not None else [
            dict(xmin=40.,ymin=30.,xmax=60.,ymax=50.)]}))
    def append(self, seq):
        frame=fixtures.make_frame(seq,10.+seq*.2)
        self.policy.append(frame,now=frame.candidates_ready_monotonic_s+.01)
        return frame
    def test_done_before_timeout_but_late_poll_is_valid_transport(self):
        self.policy.close();self.policy=self.make_policy(response_timeout_s=2.)
        self.source();self.reply(12.)
        for seq in range(1,12):current=self.append(seq)
        result=self.policy.poll(current,now=12.23)
        self.assertEqual(result['status'],'SEMANTIC_LOCAL_ASSOCIATION_VALID')
        trace=result['terminal_request_trace'][0]
        self.assertAlmostEqual(trace['transport_completion_elapsed_s'],1.98)
        self.assertAlmostEqual(trace['completed_to_poll_delay_s'],.23)
    def test_completed_transport_and_expired_source_are_separate(self):
        self.source();self.reply(10.8)
        for seq in range(1,42):current=self.append(seq)
        result=self.policy.poll(current,now=18.23)
        self.assertEqual(result['reason'],'SOURCE_CHAIN_AGE_EXPIRED')
        self.assertNotEqual(result['reason'],'RESPONSE_TIMEOUT')
    def test_unfinished_request_keeps_only_slot_after_timeout(self):
        self.source()
        for seq in range(1,52):current=self.append(seq)
        result=self.policy.poll(current,now=20.23)
        self.assertEqual(result['reason'],'RESPONSE_TIMEOUT')
        self.assertEqual(self.policy.request(current,now=20.24)['reason'],'TRANSPORT_SLOT_OCCUPIED')
        self.assertEqual(len(self.executor.calls),1)
    def test_future_completion_after_transport_budget_rejected(self):
        self.policy.close();self.policy=self.make_policy(response_timeout_s=2.)
        self.source();self.reply(12.1)
        for seq in range(1,12):current=self.append(seq)
        self.assertEqual(self.policy.poll(current,now=12.23)['reason'],'RESPONSE_TIMEOUT')
    def test_missing_actual_frame_breaks_old_request_but_allows_new_source(self):
        self.source()
        current=fixtures.make_frame(2,10.4)
        self.assertEqual(self.policy.poll(current,now=10.42)['reason'],'CHAIN_SEQUENCE_NOT_CONTIGUOUS')
        self.reply(10.5)
        current=self.append(3)
        self.assertEqual(self.policy.poll(current,now=10.63)['reason'],'CHAIN_SEQUENCE_NOT_CONTIGUOUS')
        current=fixtures.make_frame(4,13.2)
        self.policy.poll(current,now=13.22)
        self.assertEqual(self.policy.request(current,now=13.23)['status'],'SUBMITTED')
        self.assertNotEqual(self.executor.calls[0][1].binding(),self.executor.calls[1][1].binding())
    def test_gap_over_half_second_does_not_repair_old_route(self):
        self.source();current=fixtures.make_frame(1,10.6)
        self.policy.append(current,now=10.62);self.reply(10.7)
        result=self.policy.poll(fixtures.make_frame(2,10.8),now=10.82)
        self.assertEqual(result['reason'],'CHAIN_LINK_GAP_EXCEEDED')
    def test_occluded_route_cannot_restart_when_boxes_return(self):
        self.source();empty=replace(fixtures.make_frame(1,10.2),candidates=())
        self.policy.append(empty,now=10.22);current=self.append(2);self.reply(10.35)
        self.assertEqual(self.policy.poll(current,now=10.43)['reason'],'CHAIN_NO_UNIQUE_MATCH')
    def test_ambiguous_intermediate_frame_rejects_even_if_latest_unique(self):
        self.source();frame=fixtures.make_frame(1,10.2)
        frame=replace(frame,candidates=frame.candidates+(Candidate('ambiguous',(40.,30.,60.,50.),.9,'balloon'),))
        self.policy.append(frame,now=10.22);current=self.append(2);self.reply(10.35)
        self.assertEqual(self.policy.poll(current,now=10.43)['reason'],'CHAIN_ASSOCIATION_AMBIGUOUS')
    def test_source_candidate_merge_reverse_ambiguity_is_rejected(self):
        source=fixtures.make_frame(0,10.)
        source=replace(source,candidates=source.candidates+(Candidate('prior-rival',(45.,30.,65.,50.),.9,'balloon'),))
        self.policy.request(source,now=10.02)
        self.reply(10.15,boxes=[dict(xmin=40.,ymin=30.,xmax=50.,ymax=50.)])
        self.assertEqual(self.policy.poll(fixtures.make_frame(1,10.2),now=10.22)['reason'],'CHAIN_ASSOCIATION_AMBIGUOUS')
    def test_context_reset_wins_over_completed_old_reply(self):
        self.source();self.reply(10.15)
        self.policy.reset(session_id='surface-session',target_generation=3,description='new description',now=10.16)
        frame=replace(fixtures.make_frame(1,10.2),target_generation=3)
        self.assertEqual(self.policy.poll(frame,now=10.22)['reason'],'CONTEXT_RESET')
    def test_bad_binding_or_multiple_boxes_never_produce_identity(self):
        self.source();binding=dict(self.executor.calls[-1][1].binding(),image_sha256='0'*64)
        self.reply(10.15,binding=binding)
        self.assertEqual(self.policy.poll(fixtures.make_frame(1,10.2),now=10.22)['reason'],'RESPONSE_BINDING_MISMATCH')
    def test_closed_reply_is_retained_only_as_post_close_diagnostic(self):
        self.source();self.reply(10.15);self.clock_value=10.3
        receipt=self.policy.close(wait=True)
        self.assertFalse(receipt['transport_still_running'])
        self.assertEqual(receipt['drain']['status'],'TRANSPORT_DRAINED_NO_RESPONSE_REUSE')
        self.assertEqual(self.policy.poll(fixtures.make_frame(1,10.2),now=10.31)['reason'],'POLICY_CLOSED')


if __name__=='__main__':unittest.main()

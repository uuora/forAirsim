import unittest
from dataclasses import replace
import test_t1_candidate_surface_v3 as fixtures
from fixed_map_delayed_identity import ReviewHandoffTracker
from t1_delayed_semantic_bridge import DelayedSemanticPolicy,verify_delayed_selection
from t1_semantic_perception import SemanticReply

class HandoffTests(unittest.TestCase):
    def test_new_review_waits_for_three_new_frames_without_extending_old_source(self):
        executor=fixtures.ManualExecutor();policy=DelayedSemanticPolicy(lambda r:None,session_id='surface-session',target_generation=2,description=fixtures.DESCRIPTION,executor=executor)
        tracker=ReviewHandoffTracker('surface-session',2,fixtures.DESCRIPTION);initial=fixtures.make_frame(0,10.)
        policy.request(initial,now=10.02);future,request=executor.calls[-1]
        future.set_result(SemanticReply(request.binding(),10.1,dict(result_bbox_list=[dict(xmin=40.,ymin=30.,xmax=60.,ymax=50.)])))
        old_anchor=None;new_anchor=None;handoffs=[]
        for seq in range(1,37):
            frame=fixtures.make_frame(seq,10+seq*.2);now=10+seq*.2+.03
            if seq==32:
                future,request=executor.calls[-1];new_anchor=request.request_id
                future.set_result(SemanticReply(request.binding(),16.41,dict(result_bbox_list=[dict(xmin=40.,ymin=30.,xmax=60.,ymax=50.)])))
            semantic=policy.poll(frame,now=now)
            result=tracker.update(frame,semantic if semantic['status']=='SEMANTIC_LOCAL_ASSOCIATION_VALID' else None,now)
            if seq==15:
                old_anchor=result['semantic_anchor'];self.assertEqual(policy.request(frame,now=now+.01)['status'],'SUBMITTED')
            if seq>=4:
                self.assertEqual(result['state'],'TRACKING');verify_delayed_selection(frame,result,fixtures.DESCRIPTION,now)
            if seq in [32,33]:self.assertEqual(result['semantic_anchor'],old_anchor)
            if result.get('review_handoff'):handoffs.append(result['review_handoff'])
        self.assertEqual(len(handoffs),1);self.assertEqual(result['semantic_anchor'],new_anchor)
        self.assertEqual(handoffs[0]['confirmed_fresh_frames'],3);policy.close();tracker.close()
    def test_occlusion_drops_live_chain_and_cannot_reuse_same_anchor(self):
        executor=fixtures.ManualExecutor();policy=DelayedSemanticPolicy(lambda r:None,session_id='surface-session',target_generation=2,description=fixtures.DESCRIPTION,executor=executor)
        tracker=ReviewHandoffTracker('surface-session',2,fixtures.DESCRIPTION)
        policy.request(fixtures.make_frame(0,10.),now=10.02);future,request=executor.calls[-1]
        future.set_result(SemanticReply(request.binding(),10.1,dict(result_bbox_list=[dict(xmin=40.,ymin=30.,xmax=60.,ymax=50.)])))
        for seq in range(1,6):
            frame=fixtures.make_frame(seq,10+seq*.2)
            if seq==4:frame=replace(frame,candidates=())
            semantic=policy.poll(frame,now=10+seq*.2+.03)
            result=tracker.update(frame,semantic if semantic['status']=='SEMANTIC_LOCAL_ASSOCIATION_VALID' else None,10+seq*.2+.03)
            if seq>=4:self.assertNotEqual(result['state'],'TRACKING')
        policy.close();tracker.close()

if __name__=='__main__':unittest.main()

"""Bounded review handoff; keep only a still-valid old chain while confirming a new review.

No timestamp extension. Three fresh confirmations required before handoff.
"""
from t1_delayed_semantic_bridge import DelayedChainIdentityTracker

class ReviewHandoffTracker:
    def __init__(self,session_id,generation,description):
        self.context=(session_id,generation,description);self.active=DelayedChainIdentityTracker(*self.context);self.pending=None;self.pending_anchor=None;self.committed_anchor=None;self.config=self.active.config
    def update(self,frame,semantic_decision=None,now=None):
        new=isinstance(semantic_decision,dict) and semantic_decision.get('status')=='SEMANTIC_LOCAL_ASSOCIATION_VALID'
        anchor=semantic_decision.get('delayed_chain_proof',{}).get('request_binding',{}).get('request_id') if new else None
        if self.committed_anchor is None:
            result=self.active.update(frame,semantic_decision,now)
            if result.get('state')=='TRACKING':self.committed_anchor=result['semantic_anchor']
            return result
        if new and anchor not in [self.committed_anchor,self.pending_anchor]:
            self.pending=DelayedChainIdentityTracker(*self.context);self.pending_anchor=anchor
        old=self.active.update(frame,semantic_decision if anchor==self.committed_anchor else None,now)
        if self.pending is not None:
            candidate=self.pending.update(frame,semantic_decision if anchor==self.pending_anchor else None,now)
            if candidate.get('state')=='TRACKING':
                if old.get('state')=='TRACKING' and old.get('selected_candidate_id')!=candidate.get('selected_candidate_id'):
                    self.active.close();self.pending.close();self.active=DelayedChainIdentityTracker(*self.context);self.pending=None;self.pending_anchor=None;self.committed_anchor=None
                    return dict(old,status='UNCERTAIN',state='AMBIGUOUS',reason='OLD_NEW_REVIEW_CURRENT_SELECTION_CONFLICT',selected_candidate_id=None)
                previous=self.committed_anchor;self.active=self.pending;self.pending=None;self.committed_anchor=self.pending_anchor;self.pending_anchor=None
                return dict(candidate,review_handoff=dict(previous_anchor=previous,current_anchor=self.committed_anchor,confirmed_fresh_frames=3,old_source_age_never_extended=True))
        if old.get('state') in ['LOST','AMBIGUOUS'] and self.pending is None:self.committed_anchor=None
        return old
    def close(self):
        self.active.close()
        if self.pending:self.pending.close()

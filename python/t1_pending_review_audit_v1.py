"""Audit retained review confirmation separately from the next HTTP request."""
from copy import deepcopy
from t1_delayed_semantic_bridge import (
    _frame_from_dict, _hop, _node, DelayedBridgeConfig, verify_delayed_chain_proof,
)


def pending_review_audit(report):
    frames=report.get('acquisition_frames',[])+report.get('frames',[])
    saved={f['frame_id']:f.get('observer_result',{}).get('semantic_frame') for f in frames}
    terminal=report.get('observer_close',{}).get('request_terminal_trace',{}).get('terminal_request_trace',[])
    accepted={t['request_binding']['request_id']:t for t in terminal if t.get('status')=='ACCEPTED_CHAIN'}
    retained={};rows=[];issues=[]
    for f in frames:
        observer=f.get('observer_result',{});poll=observer.get('semantic_poll',{})
        selection=observer.get('temporal_identity',{});handoff=selection.get('review_handoff',{})
        if poll.get('status')=='SEMANTIC_LOCAL_ASSOCIATION_VALID':
            proof=poll.get('delayed_chain_proof',{})
            rid=proof.get('request_binding',{}).get('request_id')
            if rid:retained[rid]=deepcopy(proof)
        pending=handoff.get('pending_request_id');active=handoff.get('active_request_id')
        if selection.get('status')!='TEMPORAL_LOCAL_ASSOCIATION_VALID' or not pending or active==pending:continue
        bad=[];audit=None;proof=deepcopy(retained.get(pending,{}))
        try:
            if not proof:raise ValueError('RETAINED_PENDING_REVIEW_SOURCE_NOT_SAVED')
            frame=_frame_from_dict(observer['semantic_frame'])
            previous=_frame_from_dict(proof['nodes'][-1]['frame'])
            if previous.frame_id!=frame.frame_id:
                candidate=next(c for c in previous.candidates if c.candidate_id==proof['nodes'][-1]['selected_candidate_id'])
                candidate,link=_hop(previous,candidate,frame,DelayedBridgeConfig(**proof['configuration']))
                proof['nodes'].append(_node(frame,candidate,link))
                adopted=proof['adoption']['accepted_frame_sequence']
                proof['post_adoption_frame_ids']=[n['frame']['frame_id'] for n in proof['nodes'] if n['frame']['frame_sequence']>adopted]
            now=observer['process_completed_host_monotonic_s']
            audit=verify_delayed_chain_proof(proof,current_frame=frame,description=report['configuration']['description'],now=now)
            anchor=accepted.get(pending,{})
            if (anchor.get('request_binding')!=proof['request_binding'] or any(anchor.get(k)!=proof[k]
                    for k in ('request_sha256','semantic_reply_sha256','reply_result_sha256'))):
                raise ValueError('RETAINED_PENDING_REVIEW_NOT_BOUND_TO_REAL_ACCEPTED_REQUEST')
            if any(saved.get(n['frame']['frame_id'])!=n['frame'] for n in proof['nodes']):
                raise ValueError('PENDING_REVIEW_NODE_NOT_BOUND_TO_ACTUAL_SAVED_FRAME')
            if (audit['selected_candidate']!=selection.get('selected_candidate')
                    or min(3,1+audit['post_adoption_confirmation_count'])!=handoff.get('confirmation_count')):
                raise ValueError('PENDING_REVIEW_CURRENT_CANDIDATE_OR_CONFIRMATION_DIFFERS')
            retained[pending]=proof
        except (ValueError,TypeError,KeyError,StopIteration) as error:
            bad.append(str(error) if isinstance(error,ValueError) else 'PENDING_REVIEW_INPUT_INVALID')
        rows.append(dict(frame_id=f['frame_id'],pending_request_id=pending,
            transport_request_id=poll.get('request_binding',{}).get('request_id'),
            transport_and_pending_review_ids_differ=pending!=poll.get('request_binding',{}).get('request_id'),
            status='PASS_SAVED_PENDING_REVIEW_CHAIN' if not bad else 'FAIL_SAVED_PENDING_REVIEW_CHAIN',
            proof_audit=audit,issues=bad))
        issues.extend(dict(frame_id=f['frame_id'],issue=x) for x in bad)
    return dict(status='PASS_SAVED_PENDING_REVIEW_AUDIT' if not issues else 'FAIL_SAVED_PENDING_REVIEW_AUDIT',
        rows=rows,issues=issues,scope='bound retained accepted review plus actual current-frame hops; no future frame or source clock rebasing')

"""Freeze delayed-chain initial acquisition and unchanged measured flight phases.

Preparation makes no simulator, model or network calls and copies only the
explicit project sources and artifacts inherited from V2. Credentials are
excluded from the inventory.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from prepare_t1_identity_flight_v2 import prepare_async as prepare_v2
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified, phase_grid


NEW_SOURCES=(
    'python/t1_identity_flight_v3.py',
    'python/prepare_t1_identity_flight_v3.py',
    'scripts/run-t1-identity-flight-v3.ps1',
    'python/test_t1_identity_flight_v3.py',
    'python/t1_delayed_semantic_bridge.py',
    'python/t1_delayed_semantic_observer.py',
    'python/t1_candidate_surface_v3.py',
)


def prepare_async(output,mode,session_id,device='cuda',delay_s=0.):
    if any(not (ROOT/name).is_file() for name in NEW_SOURCES):
        raise ValueError('V3_DECLARED_SOURCE_MISSING')
    plan=prepare_v2(output,mode,session_id,device,delay_s)
    output=Path(output).resolve()
    for name in NEW_SOURCES:
        source=ROOT/name;sha=digest(source);snapshot=output/'source_bundle'/name
        copy_verified(source,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(source),sha256=sha,
                                             snapshot_path=str(snapshot)))
        plan['source_sha256'][name]=sha
    plan.update(
        control_architecture='INDEPENDENT_CONTROL_CHAIN_REVIEW_V3',
        semantic_strategy='ORIGINAL_SOURCE_BOUNDED_DELAYED_LOCAL_CHAIN_WITH_POST_ADOPTION_CONFIRMATION',
        semantic_source_chain_max_age_s=8.,
        semantic_transport_timeout_s=10.,
        local_chain_max_gap_s=.5,
        semantic_post_adoption_confirmation_frames=2,
        initial_identity_acquisition=dict(
            maximum_duration_s=20.,period_s=.4,maximum_planned_slots=50,
            desired_xy_m_s=[0.,0.],
            success='CURRENT_TRACKING_SELECTION_AND_PAIRED_SURFACE_BEFORE_20S',
            measured_epoch='START_ONLY_AFTER_SUCCESS',
            failure='FAIL_WITH_ALL_MEASURED_SLOTS_NOT_ENTERED_RETAINED',
            unused_slots='NOT_REQUIRED_AFTER_ACQUISITION',
            scope='Real onboard current frames; separate from measured phase denominator'),
        planned_acquisition_slots=phase_grid([dict(phase_id='initial_acquisition',duration_s=20.)],.4),
        global_frame_sequence_scope='Consumed acquisition slots then measured slots; missed slots consume indices, unused acquisition tail does not',
        timing_results_scope='Acquisition and measured phases separately retained; current RGB 0.5s, own state 0.12s, lease 0.12s and XY cap 0.03m/s unchanged',
    )
    plan['observer_configuration'].update(
        semantic_max_age_s=8.,semantic_source_chain_max_age_s=8.,response_timeout_s=10.,
        transport_timeout_s=10.,track_max_unreviewed_s=8.,track_max_gap_s=.5,
        confirmation_frames=2,post_adoption_confirmation_frames=2,
        current_frame_max_age_s=.5,
    )
    save(output/'plan.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('zero','short'),required=True)
    parser.add_argument('--session-id',required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--delay-s',type=float,default=0.)
    args=parser.parse_args()
    plan=prepare_async(args.output,args.mode,args.session_id,args.device,args.delay_s)
    print(plan['status'],plan['control_architecture'],len(plan['planned_acquisition_slots']),
          len(plan['planned_slots']))

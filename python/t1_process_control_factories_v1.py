"""Public geometry factory; construction happens on the child RPC owner."""
from t1_semantic_perception import Candidate, SemanticFrame


def certificate_frame(value):
    keys = {'session_id', 'target_generation', 'frame_id', 'frame_sequence', 'image_sha256',
            'width', 'height', 'readback_receipt_monotonic_s', 'candidates_ready_monotonic_s', 'candidates'}
    if set(value) != keys:
        raise ValueError('EXACT_CERTIFICATE_FRAME_REQUIRED')
    copied = dict(value)
    copied['candidates'] = tuple(Candidate(**dict(c, bbox_xyxy_pixels=tuple(c['bbox_xyxy_pixels'])))
                                for c in copied['candidates'])
    return SemanticFrame(**copied)


def make_airsim(config, log):
    import airsim
    from t1_feedback_dispatch_v1 import TimedFeedbackFlight
    from t1_identity_motion_policy import MotionPolicyConfig
    from t1_identity_motion_policy_v1p1 import IdentityMotionPolicyV1P1
    client = airsim.MultirotorClient(timeout_value=2.)
    flight = TimedFeedbackFlight(client, config['translation'], config['world_z'],
                                config['low'], config['high'], log)
    flight.stage = 'PERCEPTION_XY'
    policy = IdentityMotionPolicyV1P1(config['session_id'], 1, config['description'],
        config=MotionPolicyConfig(workspace_low_ned_m=tuple(config['low']),
                                 workspace_high_ned_m=tuple(config['high'])))
    return flight, policy

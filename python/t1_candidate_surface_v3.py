"""Strict delayed-chain selection to the unchanged paired-depth geometry.

The original RGB source and reply remain bound to their original eight-second
deadline. A late semantic reply requires a complete local association chain and
two actual newer frames after adoption before geometry is accepted. No target
truth, detector inference, network request or flight command enters this module.
"""
from __future__ import annotations

from typing import Any

from t1_candidate_surface import SurfaceConfig, SurfaceRejection, _number
from t1_candidate_surface_v2 import (
    temporal_candidate_surface_estimate as _temporal_candidate_surface_v2,
)
from t1_delayed_semantic_bridge import verify_delayed_selection
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_semantic_perception import SemanticFrame


def temporal_candidate_surface_estimate(
        frame: SemanticFrame, *, description: str, now: float, depth,
        camera: SceneCameraModel, rgb_pose: CameraPose, depth_pose: CameraPose,
        selection_certificate: dict, pairing_contract: dict,
        config: SurfaceConfig | None = None, current_frame_max_age_s: float = .5,
        semantic_source_max_age_s: float = 8.,
        max_unreviewed_track_s: float = 8.) -> dict[str, Any]:
    """Verify the delayed selection independently before estimating geometry.

    V3 requires the bounded delayed-chain policy explicitly. It never changes
    source clocks, substitutes a fresh reply, drops proof fields, or authorizes
    motion. V2 retains its original two-second semantic-source default.
    """
    checked_now = _number(now, "CURRENT_HOST_MONOTONIC_TIME")
    current_age = _number(current_frame_max_age_s, "CURRENT_FRAME_MAX_AGE", positive=True)
    source_age = _number(semantic_source_max_age_s, "SEMANTIC_SOURCE_MAX_AGE", positive=True)
    review_age = _number(max_unreviewed_track_s, "MAX_UNREVIEWED_TRACK", positive=True)
    if current_age != .5 or source_age != 8. or review_age != 8.:
        raise SurfaceRejection("DELAYED_CHAIN_FRESHNESS_CONFIG_MISMATCH")
    try:
        audit = verify_delayed_selection(frame, selection_certificate, description, checked_now)
    except ValueError:
        # Verifier errors are closed categories. Do not propagate payload text.
        raise SurfaceRejection("DELAYED_CHAIN_SELECTION_PROOF_REJECTED") from None
    result = _temporal_candidate_surface_v2(
        frame, description=description, now=checked_now, depth=depth,
        camera=camera, rgb_pose=rgb_pose, depth_pose=depth_pose,
        selection_certificate=selection_certificate, pairing_contract=pairing_contract,
        config=config, current_frame_max_age_s=.5,
        semantic_source_max_age_s=8., max_unreviewed_track_s=8.)
    result["delayed_chain_proof_independently_checked"] = True
    result["delayed_chain_verification"] = audit
    result["selection_verifier_version"] = "t1_delayed_semantic_bridge_original_source_8s"
    return result

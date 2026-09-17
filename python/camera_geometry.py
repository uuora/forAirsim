"""Intrinsics from AirSim Scene projection, not CinemAirSim viewer FOV."""
import math

import numpy as np


def scene_intrinsics(camera_info, width, height):
    """Supports this project's centered, perspective, landscape Scene cameras.

    PIPCamera.cpp:getProjectionMatrix changes NED axes before transposing.
    For the centered perspective matrix, abs(P[0,1]) and abs(P[1,2])
    are the horizontal/vertical normalized projection scale factors.
    This is renderer metadata, not empirical lens calibration.
    """
    p = np.asarray(camera_info.proj_mat.matrix, dtype=float)
    if width <= 0 or height <= 0 or p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError("Invalid Scene projection matrix or image size")
    expected = np.zeros((4, 4))
    expected[0, 1], expected[1, 2] = p[0, 1], p[1, 2]
    expected[2, 0], expected[2, 3], expected[3, 0] = p[2, 0], p[2, 3], -1
    if (not np.allclose(p, expected, atol=1e-5) or p[0, 1] <= 0 or p[1, 2] >= 0):
        raise ValueError("Unsupported projection: expected centered AirSim perspective camera")
    fx, fy = abs(p[0, 1]) * width / 2, abs(p[1, 2]) * height / 2
    fov = math.degrees(2 * math.atan(width / (2 * fx)))
    if not math.isclose(fx, fy, rel_tol=1e-4):
        raise ValueError("Unexpected unequal pixel focal lengths; inspect camera configuration")
    return {"K_nominal": [[float(fx), 0, width/2], [0, float(fy), height/2], [0, 0, 1]],
            "horizontal_fov_deg": fov,
            "camera_info_fov_deg": float(camera_info.fov),
            "intrinsics_source": "AirSim_Scene_projection_matrix",
            "camera_info_fov_matches_scene": abs(float(camera_info.fov) - fov) < .01,
            "intrinsics_note": "centered pinhole from Scene renderer projection; not empirical calibration"}

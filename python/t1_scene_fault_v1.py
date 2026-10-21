"""Declared real AirSim occluder and distractor material changes."""
import cv2
import numpy as np
from t1_ground_perception_probe import digest


def glint_pixels():
    pixels=np.full((64,64,3),[55,35,235],np.uint8)
    pixels[12:18,20:28]=245
    pixels[12:18,52:60]=245
    return pixels


def prepare_scene_fault_assets(output):
    path=output/'small_white_patches_red_texture.png'
    if path.exists():raise FileExistsError('SCENE_FAULT_ASSET_ALREADY_EXISTS')
    if not cv2.imwrite(str(path),glint_pixels()):raise RuntimeError('SCENE_FAULT_TEXTURE_SAVE_FAILED')
    return dict(path=str(path),sha256=digest(path),image_size_wh=[64,64],
        profile='two small white rectangles; red balloon texture; no broad white band')


def apply_scene_fault(client, registry, assets, plan, phase, fault_asset, output):
    occluded=plan['source_perturbation']['condition']=='occlusion' and phase=='hidden'
    registry.ensure(client,'T1FlightOccluder',assets['1m_cube_chamfer'],
        plan['scene']['occluder_world_ned_m'] if occluded else plan['scene']['inactive_world_ned_m'],
        plan['scene']['occluder_scale_xyz'])
    glint=plan['source_perturbation']['condition']=='glint' and phase=='glint'
    path=fault_asset['path'] if glint else str(output/'plain_texture.png')
    if not client.simSetObjectMaterialFromTexture('T1FlightDistractor',path):
        raise RuntimeError('DECLARED_DISTRACTOR_TEXTURE_ASSIGNMENT_FAILED')
    return dict(occluder_active=occluded,distractor_glint_active=glint,
        applied_distractor_texture=dict(path=path,sha256=digest(path)),
        target_pose_or_texture_changed=False,
        constructor_scope='scene execution only; no scene or segmentation supplied to perception/planner')

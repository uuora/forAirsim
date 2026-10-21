"""Paired camera capture with bounded stage/type diagnostics, no retry.

Same request, pairing tolerances and readback clock as the frozen capture.
Empty/invalid input is rejected before decoding. Image bytes, exception
bodies, target actor truth and service configuration never enter diagnostics.
This module has not yet flown; V11 historical sources remain unchanged.
"""
import re
import time
import airsim
import cv2
import numpy as np
from setup_scene import vector


class CaptureRejected(RuntimeError):
    pass


def exception_identity(error):
    def safe(value):
        return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.]{1,128}',value) else 'unknown'
    return dict(exception_module=safe(type(error).__module__),exception_type=safe(type(error).__name__))


def capture(client,camera='front_center',*,log,frame_id):
    if not callable(log) or not isinstance(frame_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',frame_id):
        raise ValueError('BOUND_CAPTURE_FRAME_AND_LOG_REQUIRED')
    stage='RPC_IMAGES'
    started=time.monotonic();received=None;metadata=[]
    def emit(kind,**extra):
        log(dict(kind=kind,frame_id=frame_id,capture_stage=stage,
            recorded_monotonic_s=time.monotonic(),capture_started_host_s=started,
            capture_received_host_s=received,responses=metadata,**extra))
    def reject(reason):
        raise CaptureRejected(reason)
    try:
        emit('CAPTURE_COMPONENT_ENTER')
        responses=client.simGetImages([
            airsim.ImageRequest(camera,airsim.ImageType.Scene,False,True),
            airsim.ImageRequest(camera,airsim.ImageType.DepthPerspective,True,False),
            airsim.ImageRequest(camera,airsim.ImageType.Segmentation,False,True)],vehicle_name='DroneA')
        received=time.monotonic();stage='RESPONSE_METADATA'
        if not isinstance(responses,(list,tuple)) or len(responses)!=3:reject('CAPTURE_THREE_PAIRED_RESPONSES_REQUIRED')
        for index,response in enumerate(responses):
            width,height=int(response.width),int(response.height)
            if not 0<width<=4096 or not 0<height<=4096:reject('CAPTURE_RESPONSE_DIMENSIONS_INVALID')
            metadata.append(dict(index=index,width=width,height=height,
                png_byte_count=len(response.image_data_uint8),depth_value_count=len(response.image_data_float),
                render_readback_timestamp_ns=int(response.time_stamp)))
        emit('CAPTURE_RESPONSE_METADATA')
        rgb,depth,seg=responses
        stage='RGB_DECODE'
        if not isinstance(rgb.image_data_uint8,bytes) or not rgb.image_data_uint8:reject('CAPTURE_RGB_EMPTY_OR_INVALID_BUFFER')
        image=cv2.imdecode(np.frombuffer(rgb.image_data_uint8,np.uint8),cv2.IMREAD_COLOR)
        if image is None:reject('CAPTURE_RGB_DECODE_RETURNED_NONE')
        emit('CAPTURE_COMPONENT_COMPLETED')
        stage='SEGMENTATION_DECODE'
        if not isinstance(seg.image_data_uint8,bytes) or not seg.image_data_uint8:reject('CAPTURE_SEGMENTATION_EMPTY_OR_INVALID_BUFFER')
        segmentation=cv2.imdecode(np.frombuffer(seg.image_data_uint8,np.uint8),cv2.IMREAD_COLOR)
        if segmentation is None:reject('CAPTURE_SEGMENTATION_DECODE_RETURNED_NONE')
        emit('CAPTURE_COMPONENT_COMPLETED')
        stage='DEPTH_RESHAPE'
        if len(depth.image_data_float)!=depth.height*depth.width:reject('CAPTURE_DEPTH_LENGTH_MISMATCH')
        ranges=np.asarray(depth.image_data_float,float).reshape(depth.height,depth.width)
        if (image.shape!=segmentation.shape or image.shape[:2]!=ranges.shape
                or image.shape[:2]!=(rgb.height,rgb.width) or segmentation.shape[:2]!=(seg.height,seg.width)):
            reject('CAPTURE_PAIRED_SHAPES_MISMATCH')
        emit('CAPTURE_COMPONENT_COMPLETED')
        stage='TIMESTAMP_POSE_PAIRING'
        poses=[dict(position_local_ned_m=vector(r.camera_position),
            orientation_xyzw=[r.camera_orientation.x_val,r.camera_orientation.y_val,r.camera_orientation.z_val,r.camera_orientation.w_val],
            render_readback_timestamp_ns=int(r.time_stamp)) for r in responses]
        skew=(max(p['render_readback_timestamp_ns'] for p in poses)-min(p['render_readback_timestamp_ns'] for p in poses))/1e9
        if skew>.03 or any(p['render_readback_timestamp_ns']<=0 for p in poses):reject('CAPTURE_READBACK_TIMESTAMP_PAIRING_INVALID')
        for pose in poses[1:]:
            if (np.linalg.norm(np.array(pose['position_local_ned_m'])-np.array(poses[0]['position_local_ned_m']))>.01
                    or abs(np.dot(pose['orientation_xyzw'],poses[0]['orientation_xyzw']))<.99999):
                reject('CAPTURE_CAMERA_POSE_PAIRING_INVALID')
        emit('CAPTURE_PAIRED_COMPLETE',source_clock_rebased=False,automatic_retry=False)
        return image,ranges,segmentation,dict(capture_started_host_s=started,capture_received_host_s=received,
            capture_duration_s=received-started,rgb_depth_seg_readback_skew_s=skew,response_camera_poses=poses)
    except Exception as error:
        emit('CAPTURE_COMPONENT_FAILED',reason=str(error) if isinstance(error,CaptureRejected) else 'CAPTURE_COMPONENT_EXCEPTION',
             **exception_identity(error),automatic_retry=False)
        raise

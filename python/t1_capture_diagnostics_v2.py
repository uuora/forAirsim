"""Depth-first single RPC, preserving all original pairing rejection limits.

The local UE RenderRequest timestamps each image after its sequential GPU
readback. Reading the float depth surface first reduces the interval between
the three reported readback completions. These are not exposure timestamps.
No timestamp is rewritten, no second capture is requested, and every response
still passes V1's dimension, pose and 30 ms readback-skew checks.
"""
import airsim
from t1_capture_diagnostics_v1 import capture as original_capture

REQUEST_ORDER=(airsim.ImageType.DepthPerspective,airsim.ImageType.Scene,airsim.ImageType.Segmentation)


def capture(client,camera='front_center',*,log,frame_id):
    class DepthFirstClient:
        def simGetImages(self,requests,**kwargs):
            if [r.image_type for r in requests]!=[airsim.ImageType.Scene,airsim.ImageType.DepthPerspective,airsim.ImageType.Segmentation]:
                raise ValueError('THREE_DECLARED_IMAGE_TYPES_REQUIRED')
            ordered=[requests[1],requests[0],requests[2]]
            result=client.simGetImages(ordered,**kwargs)
            if isinstance(result,(list,tuple)) and len(result)==3:
                for request,response in zip(ordered,result):
                    if response.image_type!=request.image_type:
                        raise ValueError('DEPTH_FIRST_RESPONSE_TYPE_MISMATCH')
                return [result[1],result[0],result[2]]
            return result
    log(dict(kind='CAPTURE_REQUEST_ORDER',frame_id=frame_id,image_types=list(REQUEST_ORDER),
        returned_logical_order=[0,2,5],source_clock_rebased=False,automatic_retry=False,
        reason='local UE sequential readback; depth first; exposure simultaneity not established'))
    return original_capture(DepthFirstClient(),camera,log=log,frame_id=frame_id)

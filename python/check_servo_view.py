"""Paused rendering experiment; restores vehicle pose and pause state."""
import json
import time
from datetime import datetime
import airsim
import cv2
from setup_scene import ROOT
from target_provider import CameraTargetProvider
from balloon_detector import annotate


def main():
    c = airsim.MultirotorClient(timeout_value=10)
    if c.isApiControlEnabled(vehicle_name='DroneA'):
        raise RuntimeError('An API controller is active')
    folder = ROOT / 'logs' / datetime.now().strftime('servo_view_%Y%m%d_%H%M%S')
    folder.mkdir()
    original = c.simGetVehiclePose(vehicle_name='DroneA')
    world = c.simGetObjectPose('DroneA').position
    origin = world - original.position
    paused = c.simIsPause()
    rows = []
    try:
        c.simPause(True)
        provider = CameraTargetProvider(c)
        c.simSetCameraPose('front_center',airsim.Pose(airsim.Vector3r(1,0,-0.05),
                           airsim.Quaternionr()),vehicle_name='DroneA')
        for label, y in [('center',0.0),('offset',-0.5)]:
            local = airsim.Vector3r(-5,y,-3) - origin
            c.simSetVehiclePose(airsim.Pose(local,
                                          airsim.Quaternionr()), True, vehicle_name='DroneA')
            c.simContinueForFrames(3)
            deadline = time.monotonic()+5
            while not c.simIsPause():
                if time.monotonic()>deadline:
                    raise TimeoutError('Render frame step did not pause')
                time.sleep(0.02)
            provider.observe_with_frame('DroneA')
            observation, image, result = provider.observe_with_depth('DroneA')
            pose = c.simGetObjectPose('DroneA').position
            rows.append({'case':label,'world_ned_m':[pose.x_val,pose.y_val,pose.z_val], **observation.__dict__})
            if image is not None:
                cv2.imwrite(str(folder / (label+'.png')),annotate(image,result))
    finally:
        c.simSetVehiclePose(original,True,vehicle_name='DroneA')
        c.simPause(paused)
        (folder/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(folder)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()

"""Compare camera mounts at a paused near-target scene; restore original camera."""
import json
from datetime import datetime
import airsim
import cv2
import numpy as np
from setup_scene import ROOT, rotate
from balloon_detector import detect_balloon, annotate

c = airsim.MultirotorClient(timeout_value=5)
if not c.simIsPause():
    raise RuntimeError('Pause the scene before camera diagnostics')
vehicle = c.simGetVehiclePose(vehicle_name='DroneB')
info = c.simGetCameraInfo('front_center', vehicle_name='DroneB')
inverse = vehicle.orientation.inverse()
original = airsim.Pose(rotate(info.pose.position - vehicle.position, inverse), inverse * info.pose.orientation)
folder = ROOT/'logs'/datetime.now().strftime('near_camera_%Y%m%d_%H%M%S')
folder.mkdir()
rows = []
try:
    for label, x, z, fov in [('current', .25, -.1, 90), ('wide', .25, -.1, 120), ('forward_wide', .5, -.1, 120)]:
        c.simSetCameraPose('front_center', airsim.Pose(airsim.Vector3r(x,0,z),airsim.Quaternionr()), vehicle_name='DroneB')
        c.simSetCameraFov('front_center', fov, vehicle_name='DroneB')
        for _ in range(3):
            data=c.simGetImage('front_center',airsim.ImageType.Scene,vehicle_name='DroneB')
        image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        result=detect_balloon(image)
        cv2.imwrite(str(folder/(label+'.png')),annotate(image,result))
        rows.append(dict(label=label,x=x,z=z,fov=fov,detection=result))
finally:
    c.simSetCameraPose('front_center',original,vehicle_name='DroneB')
    c.simSetCameraFov('front_center',info.fov,vehicle_name='DroneB')
    (folder/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(folder)
    print([(r['label'],r['detection']['detected']) for r in rows])

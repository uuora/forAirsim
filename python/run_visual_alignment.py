"""Bounded lateral visual alignment; no forward motion or balloon contact."""
import json
import argparse
import time
import airsim
import cv2
import numpy as np
from run_balloon_mission import Experiment
from setup_scene import vector
from perception_state import PerceptionTracker
from visual_servo import command, stop_forward_at_range
from balloon_detector import annotate


def main(range_test=False, contact_test=False):
    run = Experiment('visual_range' if range_test else 'visual_alignment')
    c = run.client
    samples = []
    try:
        for name in run.names:
            s = c.getMultirotorState(vehicle_name=name)
            p = np.array(vector(c.simGetObjectPose(name).position))
            if c.isApiControlEnabled(vehicle_name=name) or not 0.65 < p[2] < 1.1 or np.linalg.norm(vector(s.kinematics_estimated.linear_velocity)) > 0.08:
                raise RuntimeError('Requires stationary ground vehicles with API released')
            run.timestamps[name] = c.simGetCollisionInfo(vehicle_name=name).time_stamp
        c.simSetCameraPose('front_center',airsim.Pose(airsim.Vector3r(1,0,-0.05),airsim.Quaternionr()),vehicle_name='DroneA')
        c.simPause(False)
        for name in run.names:
            c.enableApiControl(True,vehicle_name=name)
            if not c.armDisarm(True,vehicle_name=name):
                raise RuntimeError('Arming failed')
        run.phase = 'TAKEOFF'
        run.drive({n:np.array(vector(c.simGetObjectPose(n).position))*[1,1,0]+[0,0,-3] for n in run.names})
        run.phase = 'STAGING'
        run.drive(run.config.staging_world_ned_m)
        run.holds = {'DroneB':run.config.staging_world_ned_m['DroneB']}
        run.drive({'DroneA':[-5,-0.5,-3]})
        run.phase = 'VISUAL_ALIGN'
        tracker = PerceptionTracker()
        z_setpoint = c.simGetVehiclePose(vehicle_name='DroneA').position.z_val
        z_initial = z_setpoint
        # This loop records its own frames; avoid a second synchronous camera pass.
        run.last_vision_s = float('inf')
        deadline = time.monotonic()+(90 if range_test else 60)
        stable = 0
        lost = 0
        while time.monotonic()<deadline:
            c.moveByVelocityZAsync(0,0,z_setpoint,1.0,
                yaw_mode=airsim.YawMode(False,0),vehicle_name='DroneA')
            sample_started = time.monotonic()
            positions, _, _ = run.observe()
            p=positions['DroneA']
            if not (-5.5<=p[0]<=(-2.8 if range_test else -4.5) and -1.2<=p[1]<=0.5 and -3.6<=p[2]<=-2.4):
                raise RuntimeError('Visual workspace bound exceeded')
            frame_started=time.monotonic()
            obs, im, detection = (run.camera_provider.observe_with_depth('DroneA') if range_test
                                  else run.camera_provider.observe_with_frame('DroneA'))
            frame_age=time.monotonic()-frame_started
            track = tracker.update(obs.detected)
            # Alignment only: depth is not used to approach; x velocity stays zero.
            valid = obs.detected and not obs.truncated and track.state.value=='DETECTED' and frame_age<=0.5
            if range_test:
                valid = valid and obs.depth_m is not None and np.isfinite(obs.depth_m) and obs.depth_m>0
            reached = range_test and valid and obs.depth_m<=2.5
            cmd = command(valid,obs.normalised_error_xy,obs.depth_m if range_test else 3.0,obs.truncated)
            velocity = ((cmd.velocity[0] if range_test else 0),cmd.velocity[1],cmd.velocity[2])
            if reached:
                velocity=stop_forward_at_range(velocity,True)
            z_setpoint += velocity[2]*0.3
            z_setpoint = max(z_initial-0.5,min(z_initial+0.5,z_setpoint))
            lost = 0 if valid else lost+1
            if lost>5:
                raise RuntimeError('Target unavailable for six observations')
            aligned=valid and max(abs(v) for v in obs.normalised_error_xy)<=0.08
            stable = stable+1 if aligned and (not range_test or reached) else 0
            attitude = c.getMultirotorState(vehicle_name='DroneA').kinematics_estimated.orientation
            samples.append({'observation':obs.__dict__,'velocity_body_ned':velocity,'time':time.time(),
                            'world_ned_m':positions['DroneA'].tolist(),
                            'attitude_pitch_roll_yaw_rad':airsim.to_eularian_angles(attitude),
                            'sampling_seconds':time.monotonic()-sample_started})
            if im is not None:
                cv2.imwrite(str(run.folder/f'align_{len(samples):03d}.png'),annotate(im,detection))
            c.moveByVelocityZAsync(velocity[0],velocity[1],z_setpoint,duration=0.35,
                yaw_mode=airsim.YawMode(False,0),vehicle_name='DroneA').join()
            if stable>=3:
                break
        else:
            raise TimeoutError('Visual alignment timeout')
        run.report['alignment_verified'] = True
        run.report['range_stop_verified'] = bool(range_test and reached)
        if contact_test:
            if not range_test or not reached:
                raise RuntimeError('Visual contact requires a verified range stop first')
            # Reuse the tested coordinate-frame route for the last segment.
            # Visual data gates this handoff; collision remains authoritative.
            run.change('start')
            result = run.contact('DroneA')
            if not result or run.hit is None:
                raise RuntimeError('Visual handoff reached no authoritative collision')
            run.report['visual_handoff_contact_verified'] = True
        run.land()
        run.report['status'] = 'PASS'
    except BaseException as exc:
        run.report.update(status='FAIL',error=str(exc),phase=run.phase)
        raise
    finally:
        c.simPause(True)
        run.report.update(samples=samples,paused=True,
                          mode='VISUAL_HANDOFF_CONTACT' if contact_test else ('RANGE_STOP_2_5M' if range_test else 'LATERAL_ALIGNMENT_ONLY'),
                          min_separation_m=run.min_separation)
        (run.folder/'report.json').write_text(json.dumps(run.report,indent=2),encoding='utf-8')
        run.stream.close()
        print(run.folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--range',action='store_true',help='Enable bounded forward flight and stop at measured 2.5 m')
    parser.add_argument('--contact',action='store_true',help='After range stop, hand off to bounded coordinate contact')
    args=parser.parse_args()
    main(args.range,args.contact)

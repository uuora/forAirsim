"""Ground-only check of projected metadata against real colored mesh pixels."""
import json
from pathlib import Path
import time
import numpy as np
import airsim
from PIL import Image
from dual_target_control.simulator import ROOT
from dual_target_control.overview import OverviewStream

config = json.loads((ROOT / 'configs/dual_target_control.json').read_text(encoding='utf-8'))
c = airsim.MultirotorClient(port=config['airsim']['port'], timeout_value=5)
n = config['airsim']['vehicle']
s = c.getMultirotorState(vehicle_name=n)
assert s.landed_state == airsim.LandedState.Landed
assert not c.isApiControlEnabled(vehicle_name=n)
p = s.kinematics_estimated.position
yaw = airsim.to_eularian_angles(s.kinematics_estimated.orientation)[2]

class GroundState:
    def snapshot(self):
        return dict(mode='airsim', connected=True, config=config,
            position=[p.x_val, p.y_val, -p.z_val], yaw=yaw,
            targets=[[25, 25, 12], [-25, -25, 18]],
            planned_path=[[0, 0, 8], [25, 25, 12], [-25, -25, 18]], trail=[])

stream = OverviewStream(config, GroundState())
try:
    assert stream.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        snap = stream.snapshot()
        if snap['status'] == 'ERROR': raise RuntimeError(snap['message'])
        if snap['frames'] >= 4: break
        time.sleep(.1)
    else: raise TimeoutError('No stable frame')
    pixels = snap['frame'].astype(np.int16)
    masks = [(pixels[:, :, 0] > pixels[:, :, 1] + 50) & (pixels[:, :, 0] > pixels[:, :, 2] + 50),
             (pixels[:, :, 2] > pixels[:, :, 0] + 50) & (pixels[:, :, 2] > pixels[:, :, 1] + 50)]
    result = dict(scope='ground-only rendered target mesh vs metadata projection', frames=snap['frames'], rows=[])
    for expected, mask in zip(snap['overlays']['targets'], masks):
        # Exclude the blue field border and other colored scene decoration.
        # The known target sphere has a ~10 px diameter; a 30 px search window
        # still rejects a bad projection and prevents a global color centroid.
        yy, xx = np.indices(mask.shape)
        mask &= (np.abs(xx - expected[0]) < 15) & (np.abs(yy - expected[1]) < 15)
        v, u = np.nonzero(mask)
        if len(u) == 0: raise AssertionError('Target absent from real rendered frame')
        center = [float(u.mean()), float(v.mean())]
        error = float(np.linalg.norm(np.asarray(center) - expected))
        result['rows'].append(dict(projected=expected, measured_centroid=center, error_px=error, pixels=len(u)))
        assert error < 2.0
    out = ROOT / 'reports/dual_target_control'
    Image.fromarray(snap['frame']).save(out / 'overview_projection_raw.png')
    (out / 'overview_projection_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
finally:
    stream.stop()

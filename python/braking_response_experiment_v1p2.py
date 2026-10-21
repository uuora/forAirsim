"""V1.2 low-speed braking: measured acquisition and unique origin query selection."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from braking_response import entry_check
from braking_response_v1p2 import summarize_pulse
from braking_response_experiment import BrakingResponseExperiment
from braking_response_v1p1 import acquired_velocity_interval, validate_acquisition
from trajectory_single_experiment import ProtectionRequired


class AcquiredBrakingExperiment(BrakingResponseExperiment):
    def __init__(self, config_path, client=None):
        super().__init__(config_path, client=client)
        self.measurement = validate_acquisition(self.cfg)
        self.report['measurement_protocol_version'] = '1.2_unique_origin_query'
        self.save()

    def run_pulse(self, spec, index):
        pulse = dict(spec, pulse_id=f'pulse_{index:03d}', status='PREPARING', motion_request_started=False, samples=[])
        self.report['pulses'].append(pulse); self.current_pulse=pulse; self.save(); self.prepare()
        pulse['status']='RAMPING'; desired=np.array(spec['direction_xy'])*spec['amplitude_m_s']
        sample=self.query('RAMP'); start=time.monotonic()
        while np.linalg.norm(self.active_xy_command-desired)>1e-9:
            if time.monotonic()-start>self.measurement['ramp_timeout_s']: raise ProtectionRequired('RAMP_TIMEOUT')
            self.dispatch(desired,sample,'RAMP'); sample=self.query('RAMP')
        pulse['status']='ACQUIRING_ENTRY_VELOCITY'; start=time.monotonic(); acquired=[]
        pulse['entry_acquisition']=dict(started_host_s=start, timeout_s=self.measurement['entry_acquisition_timeout_s'], status='RUNNING')
        while time.monotonic()-start<self.measurement['entry_acquisition_timeout_s']:
            self.dispatch(desired,sample,'ACQUIRE'); sample=self.query('ACQUIRE'); acquired.append(sample)
            result=acquired_velocity_interval(acquired,spec,self.measurement)
            if result['accepted']:
                pulse['entry_acquisition'].update(result, status='ACCEPTED', finished_host_s=time.monotonic()); break
        else:
            pulse['entry_acquisition'].update(status='TIMEOUT', finished_host_s=time.monotonic())
            raise ProtectionRequired('ENTRY_VELOCITY_ACQUISITION_TIMEOUT')
        pulse['status']='HOLDING'; start=time.monotonic()
        while time.monotonic()-start<self.measurement['hold_s']:
            self.dispatch(desired,sample,'HOLD'); sample=self.query('HOLD')
        pulse['entry_check']=entry_check(sample,spec,self.measurement)
        if not pulse['entry_check']['accepted']: raise ProtectionRequired('BRAKE_ENTRY_VELOCITY_NOT_ACCEPTED')
        pulse['status']='BRAKING'; start=time.monotonic(); stable=None
        while time.monotonic()-start<self.measurement['brake_timeout_s']:
            self.dispatch([0.,0.],sample,'BRAKE'); sample=self.query('BRAKE')
            stopped=(np.linalg.norm(self.active_xy_command)<=1e-9 and sample['speed_norm_m_s']<=self.measurement['stop_speed_m_s']
                and sample['acceleration_norm_m_s2']<=self.measurement['stop_acceleration_m_s2'])
            stable=(sample['host_monotonic_s'] if stable is None else stable) if stopped else None
            if stable is not None and sample['host_monotonic_s']-stable>=self.measurement['stop_stable_s']:
                pulse['status']='COMPLETED'; break
        else: raise ProtectionRequired('BRAKE_STRICT_STABILITY_TIMEOUT')
        pulse['diagnostics']=summarize_pulse(pulse,self.measurement,self.model())
        if not pulse['diagnostics']['counts_as_measured_stopping']: raise ProtectionRequired('MEASUREMENT_COVERAGE_OR_STABILITY_FAILED')
        self.current_pulse=None; self.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--receipt-path',type=Path,required=True); args=parser.parse_args()
    run=AcquiredBrakingExperiment(args.config)
    if args.receipt_path.exists(): raise ValueError('Trial pointer must not overwrite an existing file')
    args.receipt_path.write_text(json.dumps({'report_path':str(run.folder/'report.json'),
        'created_local':datetime.now().astimezone().isoformat()},indent=2),encoding='utf-8')
    run.run(); raise SystemExit(0 if run.report['measurement_passed'] else 2)

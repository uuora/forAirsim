"""Zero gate confirmation spans saved queries; inherited pulse logic unchanged."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from braking_response_experiment_v1p2 import AcquiredBrakingExperiment
from trajectory_single_experiment import ProtectionRequired


class SavedSpanBrakingExperiment(AcquiredBrakingExperiment):
    def approach(self):
        if self.measurement['mode']!='zero_hold_gate':return super().approach()
        self.report['zero_gate_time_basis']='span_of_saved_query_receipts'
        try:
            self.prepare();sample=self.query('ZERO_HOLD');first=sample['host_monotonic_s'];start=time.monotonic()
            while sample['host_monotonic_s']-first<self.measurement['stop_stable_s']:
                if time.monotonic()-start>self.measurement['brake_timeout_s']:raise ProtectionRequired('ZERO_QUERY_SPAN_TIMEOUT')
                self.dispatch([0.,0.],sample,'ZERO_HOLD');sample=self.query('ZERO_HOLD')
            self.report['zero_gate_saved_query_span_s']=sample['host_monotonic_s']-first
            self.report['measurement_passed']=all(not s['protection_reasons'] and
                s['speed_norm_m_s']<=self.cfg['initial_state_match']['speed_tolerance_m_s'] for s in self.report['zero_gate_samples'])
        except (ProtectionRequired,ValueError) as error:self.report['protection_reason']=str(error)
        finally:
            self.client.cancelLastTask(vehicle_name=self.vehicle);self.client.hoverAsync(vehicle_name=self.vehicle).join();self.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--receipt-path',type=Path,required=True);args=parser.parse_args();run=SavedSpanBrakingExperiment(args.config)
    if args.receipt_path.exists():raise ValueError('Trial pointer must not overwrite existing file')
    args.receipt_path.write_text(json.dumps({'report_path':str(run.folder/'report.json'),
        'created_local':datetime.now().astimezone().isoformat()},indent=2),encoding='utf-8')
    run.run();raise SystemExit(0 if run.report['measurement_passed'] else 2)

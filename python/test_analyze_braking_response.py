import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from analyze_braking_response import audit_report, audit_sample, audit_commands, equal
from trajectory_execution_guard import boundary_command
from trajectory_timed_reference import command_ramp_interval


def fixture():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/braking_response_v1/zero_short.json').read_text(encoding='utf-8'))
    def sample(t):
        return dict(world_ned_m=[-5.,-.5,-3.], velocity_ned_m_s=[0.,0.,0.], acceleration_ned_m_s2=[0.,0.,0.],
            actual_body_rpy_deg=[0.,0.,0.], speed_norm_m_s=0., acceleration_norm_m_s2=0., workspace_margin_m=.6,
            height_error_m=0., protection_reasons=[], over_planning_speed_budget=False, over_planning_acceleration_budget=False,
            host_monotonic_s=t, query_started_host_s=t, query_state_timestamp_ns=int(t*1e9), stage='ZERO_HOLD',
            previous_command_velocity_xy_m_s=[0.,0.])
    samples = [sample(2.+i*.2) for i in range(4)]; commands = []; events = []; cfg = config['planning']
    for i, origin in enumerate(samples[:-1]):
        t = origin['host_monotonic_s']; timing = command_ramp_interval(None if i == 0 else commands[-1]['dispatch_host_s'], t, .2)
        env = boundary_command(origin['world_ned_m'], [0.,0.], [0.,0.], np.array(config['workspace_min_ned_m'])+.15,
            np.array(config['workspace_max_ned_m'])-.15, .15, .15, max(timing['effective_interval_s'],1e-12), .5, .02)
        row = dict(sequence=i, stage='ZERO_HOLD', pulse_id=None, desired_xy_m_s=[0.,0.], command_velocity_xy_m_s=[0.,0.],
            previous_command_velocity_xy_m_s=[0.,0.], ramp_timing=timing, boundary_envelope=env, origin_sample=origin,
            request_intent_host_s=t, dispatch_host_s=t+.001, rpc_return_host_s=t+.06, requested_duration_s=.05,
            status='RPC_COMPLETED', target_used=False)
        intent = {k:v for k,v in row.items() if k not in ('dispatch_host_s','rpc_return_host_s')}; intent['status']='DISPATCHING'
        commands.append(row); events.extend([intent, row])
    accepted = []
    for i in range(4):
        row = sample(1.+i*.2); row['accepted']=True; accepted.append(row)
    match = dict(status='ACCEPTED', specification=config['initial_state_match'], desired_world_ned_m=[-5.,-.5,-3.],
        samples=accepted, accepted_state=accepted[-1])
    raw = dict(configuration=config, formal_thesis_effect_results=False, target=None, truth_access=dict(target_pose='NOT_READ', RGB_depth='NOT_READ'),
        zero_gate_samples=samples, measurement_commands=commands, initial_state_matches=[match], pulses=[],
        status='PASS', paused=True, measurement_passed=True, flight_dispatch_started=True,
        landing_evidence=dict(landed_flag=True, api_control_released=True, stable_ground_seconds=2.1, speed_m_s=0.,
            world_ned_m=[-5.,-4.,.8], ground_reference_z_m=.8, ground_actor='Ground_6'))
    return raw, config, events


def write_ledgers(folder, raw, events):
    for name, rows in [('measurement_samples.jsonl', raw['zero_gate_samples']),
                       ('measurement_commands.jsonl', raw['measurement_commands']), ('measurement_command_events.jsonl', events)]:
        (folder/name).write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


class BrakingAuditTests(unittest.TestCase):
    def test_valid_replay_and_physical_landing(self):
        raw, cfg, events = fixture()
        with TemporaryDirectory() as name:
            folder = Path(name); write_ledgers(folder, raw, events); result = audit_report(raw, cfg, folder)
            self.assertTrue(result['gate_passed'], result['errors'])
            raw['landing_evidence']['world_ned_m']=[-5.,-4.,-3.]
            self.assertFalse(audit_report(raw, cfg, folder)['safe_landing_verified'])

    def test_missing_completion_or_query_is_not_verified(self):
        for name in ('measurement_commands.jsonl','measurement_command_events.jsonl','measurement_samples.jsonl'):
            raw,cfg,events=fixture()
            with TemporaryDirectory() as folder:
                folder=Path(folder); write_ledgers(folder, raw, events); path=folder/name
                path.write_text('\n'.join(path.read_text().splitlines()[:-1]), encoding='utf-8')
                result=audit_report(raw,cfg,folder)
                self.assertEqual(result['integrity_status'],'INVALID'); self.assertFalse(result['gate_passed'])

    def test_changed_actual_vector_cannot_hide_behind_passing_scalar(self):
        raw,cfg,_=fixture(); sample=copy.deepcopy(raw['zero_gate_samples'][0]); sample['velocity_ned_m_s']=[.16,0,0]
        with self.assertRaises(ValueError): audit_sample(sample,cfg)

    def test_command_chain_timing_origin_and_envelope_mutations(self):
        for kind in ('future','previous','envelope','receipt','target','origin'):
            raw,cfg,_=fixture(); rows=copy.deepcopy(raw['measurement_commands'])
            if kind=='future': rows[0]['ramp_timing']['effective_interval_s']=.2
            if kind=='previous': rows[1]['previous_command_velocity_xy_m_s']=[.03,0]
            if kind=='envelope': rows[1]['command_velocity_xy_m_s']=[.03,0]
            if kind=='receipt': rows[1]['status']='DISPATCHING'
            if kind=='target': rows[1]['target_used']=True
            if kind=='origin': rows[1]['origin_sample']['world_ned_m'][0] += .01
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit_commands(rows,raw['zero_gate_samples'],cfg)

    def test_landing_success_does_not_convert_measurement_failure(self):
        raw,cfg,events=fixture(); raw['measurement_passed']=False
        with TemporaryDirectory() as name:
            folder=Path(name); write_ledgers(folder,raw,events); result=audit_report(raw,cfg,folder)
            self.assertTrue(result['safe_landing_verified']); self.assertFalse(result['gate_passed'])

    def test_nanosecond_and_boolean_equality_are_strict(self):
        self.assertFalse(equal(1791036345260408064,1791036345260408065))
        self.assertFalse(equal(True,1)); self.assertFalse(equal(float('nan'),float('nan')))


if __name__=='__main__': unittest.main()

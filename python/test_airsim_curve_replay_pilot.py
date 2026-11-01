"""Offline checks of the AirSim pilot protocol; no simulator is contacted."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from trajectory_research.curves import PolynomialSegment
from run_airsim_curve_replay_pilot import (
    Journal, initialize_curve, limit_command, load_plans, metrics_from_trace, new_execution_collision,
    paused_ground_probe, run, select_plans, validate_config, velocity_preparation_stage, zup,
)
from audit_airsim_curve_replay_pilot import audit_directory, independent_metrics


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"configs/curve_abom_airsim_pilot_20261029.json").read_text(encoding="utf-8"))


def vector(x=0., y=0., z=0.):
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


GROUP_IDS = [f'{representation}_{kind}' for kind in ('ordinary','abom','abom_improved_candidate')
             for representation in ('bezier','hermite')]


def six_plan_document():
    return dict(schema_version=1, plans=[dict(group_id=group,representation=group.split('_')[0],
        family='single_bump',parameters=[0.,1.15,0.,5.5],seed=1101,
        selection_provenance=dict(result_json=f'groups/{group}.json',model_feasible=True,model_score=.4))
        for group in GROUP_IDS])


def write_json(path, item):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_audit_fixture(folder, selected_ids, *, legacy=False, failed_first=False):
    """Synthetic record package, explicitly not AirSim execution evidence.

    Builds the full six-plan source-byte lineage independently of the producer
    selector, then writes only the requested execution slots. Trace data are
    deliberately idealized; this fixture tests evidence selection/accounting.
    """
    folder = Path(folder)
    original, evidence = folder/'synthetic_model_package', folder/'synthetic_replay'
    document = six_plan_document()
    by_id = {p['group_id']:p for p in document['plans']}
    source_plan = original/'airsim_replay_plan.json'
    write_json(source_plan, document)
    artifacts = {source_plan.name:file_sha(source_plan)}
    for p in document['plans']:
        result = dict(p, selected_validation=dict(parameters=p['parameters'],feasible=True,score=.4))
        relative = p['selection_provenance']['result_json']
        path = original/relative
        write_json(path,result)
        artifacts[relative] = file_sha(path)
    write_json(original/'completion.json',dict(status='COMPLETE',groups=6,artifact_sha256=artifacts))
    c = config()
    config_path = original/'config.json'
    write_json(config_path,c)
    evidence.mkdir()
    snapshots = evidence/'source_snapshot'
    snapshots.mkdir()
    sources = []
    for i,path in enumerate((config_path,ROOT/'python/trajectory_research/parameterizations.py',
                             ROOT/'python/trajectory_research/curves.py',source_plan)):
        target = snapshots/f'{i:02d}_{path.name}'
        target.write_bytes(path.read_bytes())
        sources.append(dict(path=str(path),snapshot=target.relative_to(evidence).as_posix(),sha256=file_sha(path)))
    probe = dict(loaded_settings=c['settings'],vehicles=['ResearchDrone'],status='PASS',
                 checks=dict(synthetic_fixture=True),armed_by_probe=False,movement_commands_sent=0,
                 step=dict(observed_sim_s=.009))
    write_json(evidence/'ground_probe.json',probe)
    reports = []
    for group in selected_ids:
        p = by_id[group]
        group_dir = evidence/group
        group_dir.mkdir()
        if failed_first:
            report = dict(group_id=group,plan=p,status='FAILED',state_samples=0,command_requests=0,
                          failure_type='SyntheticInitializationError',failure_reason='Synthetic initialization failure')
            write_json(group_dir/'trace.json',dict(states=[],commands=[]))
            (group_dir/'events.jsonl').write_text('',encoding='utf-8')
            write_json(group_dir/'report.json',report)
            reports.append(report)
            break
        from trajectory_research.parameterizations import curve_from_parameters
        curve = curve_from_parameters(p['parameters'],family=p['family'],representation=p['representation'])
        t0 = 10_000_000_000
        timestamps = [t0+i*50_000_000 for i in range(111)]
        times = [(t-t0)*1e-9 for t in timestamps]
        states = []
        for i,t in enumerate(times):
            state = dict(sim_timestamp_ns=timestamps[i],reference_elapsed_s=t,reference_sample_time_s=t,
                reference_position_zup_m=curve.sample(t).tolist(),reference_velocity_zup_mps=curve.sample(t,1).tolist(),
                position_zup_m=curve.sample(t).tolist(),velocity_zup_mps=curve.sample(t,1).tolist(),
                read_requested_host_s=float(i)+.02,read_returned_host_s=float(i)+.021)
            if i:
                state.update(collision=dict(has_collided=False,time_stamp=0),new_execution_collision=False)
            states.append(state)
        commands = []
        previous = curve.sample(0.,1)
        for i,t in enumerate(times[:-1]):
            dt = .02 if i == 0 else times[i]-times[i-1]
            raw = curve.sample(t,1)
            raw_norm = float(np.linalg.norm(raw))
            speed = raw*min(1.,4./max(raw_norm,1e-300))
            delta = speed-previous
            dn = float(np.linalg.norm(delta))
            limited = previous+delta*min(1.,4.*dt/max(dn,1e-300))
            commands.append(dict(command_index=i,read_sim_timestamp_ns=timestamps[i],reference_elapsed_s=t,
                ramp_interval_s=dt,raw=raw.tolist(),speed_limited=speed.tolist(),limited=limited.tolist(),
                speed_clipped=raw_norm>4.,slew_clipped=dn>4.*dt,
                command_slew_mps2=float(np.linalg.norm(limited-previous)/dt),
                velocity_command_ned_mps=(limited*np.array([1.,1.,-1.])).tolist(),
                requested_command_duration_s=c['controller']['command_duration_s'],
                send_requested_host_s=i+1.,async_returned_host_s=i+1.001,
                step_started_host_s=i+1.002,step_returned_host_s=i+1.019,
                actual_state_gap_sim_s=times[i+1]-t,requested_step_wall_s=c['execution']['step_wall_request_s']))
            previous = limited
        baseline = dict(has_collided=False,time_stamp=0)
        events = [dict(kind='REFERENCE_COLLISION_BASELINE',host_s=.1,sim_start_timestamp_ns=t0,collision=baseline)]
        for i,command in enumerate(commands):
            events += [dict(kind='VELOCITY_REQUEST',host_s=i+1.001,command=command),
                       dict(kind='STATE_QUERY',host_s=i+1.022,phase='after_step',state=states[i+1]),
                       dict(kind='COMPLETED_STEP',host_s=i+1.023,state=states[i+1],command=command)]
        initialization = dict(requested_position_zup_m=curve.sample(0.).tolist(),
            requested_velocity_zup_mps=curve.sample(0.,1).tolist(),observed_state=states[0],position_error_m=0.,velocity_error_mps=0.,
            lift_rpc=dict(raw_result=True,before_state=dict(sim_timestamp_ns=1_000_000_000),
                after_state=dict(sim_timestamp_ns=4_000_000_000,landed_state=1,position_zup_m=[0.,0.,1.5]),
                observed_sim_elapsed_s=3.,rpc_elapsed_host_s=3.),
            zero_velocity_rpc=dict(raw_result=True,before_state=dict(sim_timestamp_ns=4_000_000_000),
                after_state=dict(sim_timestamp_ns=5_000_000_000),observed_sim_elapsed_s=1.,rpc_elapsed_host_s=1.))
        report = dict(group_id=group,plan=p,status='COMPLETED',state_samples=len(states),command_requests=len(commands),
            initialization=initialization,initial_collision_baseline=baseline,
            observed_collision_free=True,sampled_workspace_passed=True,post_reference=dict(synthetic_fixture=True),
            metrics=independent_metrics(curve,states,commands,c))
        write_json(group_dir/'report.json',report)
        write_json(group_dir/'trace.json',dict(states=states,commands=commands,scope='SYNTHETIC TEST FIXTURE'))
        (group_dir/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
        fields = ['sim_timestamp_ns','reference_elapsed_s','read_requested_host_s','read_returned_host_s']
        fields += [f'{prefix}_{axis}' for prefix in ('reference_p','reference_v','actual_p','actual_v') for axis in 'xyz']
        with (group_dir/'states.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for state in states:
                row={key:state[key] for key in fields[:4]}
                for prefix,key in (('reference_p','reference_position_zup_m'),('reference_v','reference_velocity_zup_mps'),
                                   ('actual_p','position_zup_m'),('actual_v','velocity_zup_mps')):
                    row.update({f'{prefix}_{axis}':value for axis,value in zip('xyz',state[key])})
                writer.writerow(row)
        reports.append(report)
    complete = not failed_first
    manifest = dict(schema_version=1,probe_only=False,config=c,plans_document=document,sources=sources,
        sources_unchanged=True,planned_replays=len(selected_ids),attempted_replays=len(reports),
        completed_replays=sum(r['status']=='COMPLETED' for r in reports),status='COMPLETED' if complete else 'FAIL',
        not_run_group_ids=selected_ids[len(reports):])
    if not legacy:
        manifest.update(selected_group_ids=list(selected_ids),unselected_group_ids=[g for g in GROUP_IDS if g not in selected_ids])
    write_json(evidence/'manifest.json',manifest)
    write_json(evidence/'rows.json',reports)
    return evidence, original


class GroundClient:
    def __init__(self, settings):
        self.settings = settings
        self.t = 1_000_000_000
        self.paused = False
        self.landed = 0
        self.operations = []
        self.kin = SimpleNamespace(position=vector(), linear_velocity=vector(), angular_velocity=vector(),
                                   orientation=SimpleNamespace(x_val=0., y_val=0., z_val=0., w_val=1.))

    def ping(self): return True
    def listVehicles(self): return ['ResearchDrone']
    def getSettingsString(self): return json.dumps(self.settings)
    def simPause(self, flag): self.paused = flag; self.operations.append(('pause', flag))
    def simIsPause(self): return self.paused
    def reset(self): self.operations.append(('reset',))
    def simGetGroundTruthKinematics(self, vehicle_name=''): return self.kin
    def simSetKinematics(self, state, ignore_collision, vehicle_name=''):
        self.operations.append(('set_kinematics',)); self.kin = state
    def getMultirotorState(self, vehicle_name=''):
        return SimpleNamespace(timestamp=self.t, landed_state=self.landed, kinematics_estimated=self.kin)
    def simContinueForTime(self, duration):
        self.operations.append(('unarmed_step', duration))
        # Deliberately differ from requested wall seconds to catch time assumptions.
        self.t += round(duration*.9*1e9)
        self.paused = True
    def enableApiControl(self, flag, vehicle_name=''):
        self.operations.append(('api_control', flag))
    def armDisarm(self, *args, **kwargs): raise AssertionError('probe must not arm')
    def moveByVelocityAsync(self, *args, **kwargs): raise AssertionError('probe must not move')


class AirSimReplayProtocolTests(unittest.TestCase):
    def test_isolated_configuration_is_enforced(self):
        validate_config(config())
        for mutate in (lambda c:c.update(port=41451), lambda c:c.update(vehicle='DroneA'),
                       lambda c:c['settings'].update(PhysicsEngineName='ExternalPhysicsEngine'),
                       lambda c:c['settings'].update(ClockType='ScalableClock')):
            c = config(); mutate(c)
            with self.assertRaises(ValueError): validate_config(c)

    def test_coordinate_conversion_is_its_own_inverse(self):
        np.testing.assert_array_equal(zup([1.,2.,-5.]), [1.,2.,5.])
        np.testing.assert_array_equal(zup(zup([1.,2.,-5.])), [1.,2.,-5.])

    def test_speed_then_slew_limit_uses_observed_ramp_interval(self):
        r = limit_command([10.,0.,0.], [0.,0.,0.], .03, 4., 4.)
        self.assertEqual(r['speed_limited'], [4.,0.,0.])
        np.testing.assert_allclose(r['limited'], [.12,0.,0.], atol=1e-15)
        self.assertTrue(r['speed_clipped']); self.assertTrue(r['slew_clipped'])
        self.assertAlmostEqual(r['command_slew_mps2'], 4.)
        with self.assertRaises(ValueError): limit_command([0,0,0], [0,0,0], 0., 4.,4.)

    def test_ground_probe_does_not_arm_or_send_movement(self):
        c = config()
        client = GroundClient(c['settings'])
        with TemporaryDirectory() as folder:
            journal = Journal(folder)
            result = paused_ground_probe(client, c, SimpleNamespace(LandedState=SimpleNamespace(Landed=0)), journal)
            journal.close()
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['armed_by_probe'])
        self.assertEqual(result['movement_commands_sent'], 0)
        self.assertAlmostEqual(result['step']['observed_sim_s'], .009, places=15)
        self.assertTrue(client.paused)
        self.assertNotIn(('api_control', True), client.operations)

    def test_probe_refuses_unrelated_vehicle(self):
        c = config(); client = GroundClient(c['settings'])
        client.listVehicles = lambda:['SomeoneElsesDrone']
        with TemporaryDirectory() as folder:
            journal = Journal(folder)
            try:
                with self.assertRaises(RuntimeError):
                    paused_ground_probe(client,c,SimpleNamespace(LandedState=SimpleNamespace(Landed=0)),journal)
            finally: journal.close()
        self.assertEqual(client.operations, [])

    def test_terminal_interpolation_preserves_nonzero_terminal_velocity(self):
        curve = PolynomialSegment([[0.,0.,5.],[1.,0.,0.]], duration_s=1.)
        times = [0., .4, .8, 1.2]
        states = [dict(reference_elapsed_s=t, position_zup_m=[t,0.,5.], velocity_zup_mps=[1.,0.,0.]) for t in times]
        commands = [dict(speed_clipped=False, slew_clipped=False) for _ in times[:-1]]
        c = config(); c['execution']['terminal_interpolation_max_gap_s'] = .5
        metrics = metrics_from_trace(curve, states, commands, c)
        self.assertEqual(metrics['position_rmse_m'], 0.)
        self.assertEqual(metrics['terminal_velocity_error_mps'], 0.)
        self.assertTrue(metrics['terminal_interpolated'])
        self.assertEqual(metrics['terminal_actual_velocity_zup_mps'], [1.,0.,0.])
        self.assertTrue(metrics['control_error_gates_passed'])

    def test_terminal_data_quality_gap_and_nonmonotonic_clock_are_rejected(self):
        curve = PolynomialSegment([[0.,0.,5.],[1.,0.,0.]], duration_s=1.)
        states = [dict(reference_elapsed_s=t, position_zup_m=[t,0.,5.], velocity_zup_mps=[1.,0.,0.]) for t in [0.,.8,1.2]]
        commands = [dict(speed_clipped=False,slew_clipped=False) for _ in range(2)]
        with self.assertRaises(ValueError): metrics_from_trace(curve,states,commands,config())
        states[1]['reference_elapsed_s'] = 0.
        with self.assertRaises(ValueError): metrics_from_trace(curve,states,commands,config())

    def test_six_frozen_plans_are_validated_before_connecting(self):
        item = dict(group_id='group_0',family='single_bump',representation='hermite',parameters=[0.,1.15,0.,5.5],seed=7)
        plans = [dict(item,group_id=f'group_{i}') for i in range(6)]
        with TemporaryDirectory() as folder:
            p = Path(folder)/'plans.json'; p.write_text(json.dumps(dict(schema_version=1,plans=plans)))
            _, actual = load_plans(p)
            self.assertEqual(len(actual),6)
            plans[1]['group_id'] = plans[0]['group_id']
            p.write_text(json.dumps(dict(plans=plans)))
            with self.assertRaises(ValueError): load_plans(p)

    def test_failed_or_void_lift_records_raw_type_and_sim_elapsed_before_failing(self):
        for raw in (False, None):
            with self.subTest(raw=raw), TemporaryDirectory() as folder:
                c = config(); client = GroundClient(c['settings'])
                client.isApiControlEnabled = lambda vehicle_name='': True
                def result():
                    client.t += 3_000_000_000
                    return raw
                lifecycle = SimpleNamespace(armDisarm=lambda *a,**k: True,
                    moveByVelocityAsync=lambda *a,**k: SimpleNamespace(get=result))
                airsim = SimpleNamespace(DrivetrainType=SimpleNamespace(MaxDegreeOfFreedom=0),YawMode=lambda *a:None)
                journal = Journal(folder)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'velocity preparation did not complete'):
                        velocity_preparation_stage(client,lifecycle,[0.,0.,.5],3.,c,airsim,journal,'fixed_vertical_lift')
                finally: journal.close()
                rows = [json.loads(s) for s in (Path(folder)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                receipt = next(r['receipt'] for r in rows if r['kind'] == 'PREPARATION_VELOCITY_RESULT')
                self.assertEqual(receipt['raw_result'], raw)
                self.assertEqual(receipt['raw_result_type'], type(raw).__name__)
                self.assertAlmostEqual(receipt['observed_sim_elapsed_s'], 3., places=13)
                self.assertEqual(receipt['after_state']['sim_timestamp_ns']-receipt['before_state']['sim_timestamp_ns'], 3_000_000_000)

    def test_v3_uniform_velocity_lift_then_zero_precedes_simulation_state_injection(self):
        c = config(); client = GroundClient(c['settings'])
        client.isApiControlEnabled = lambda vehicle_name='': True
        requests = []
        def velocity(*args,**kwargs):
            requests.append(list(args))
            def result():
                client.t += round(args[3]*1e9)
                client.kin.position = vector(0.,0.,-1.6)
                client.landed = 1
                return True
            return SimpleNamespace(get=result)
        lifecycle = SimpleNamespace(armDisarm=lambda *a,**k:True, moveByVelocityAsync=velocity)
        airsim = SimpleNamespace(DrivetrainType=SimpleNamespace(MaxDegreeOfFreedom=0),YawMode=lambda *a:None,
            LandedState=SimpleNamespace(Flying=1), KinematicsState=SimpleNamespace,
            Vector3r=vector, Quaternionr=lambda x,y,z,w:SimpleNamespace(x_val=x,y_val=y,z_val=z,w_val=w))
        curve = PolynomialSegment([[0.,0.,5.],[1.,0.,0.]],duration_s=1.)
        with TemporaryDirectory() as folder:
            journal = Journal(folder)
            try:
                observed, receipt = initialize_curve(client,lifecycle,curve,c,airsim,journal)
            finally: journal.close()
        self.assertEqual(requests, [[0.,0.,-.5,3.],[0.,0.,0.,1.]])
        self.assertEqual(observed['position_zup_m'], [0.,0.,5.])
        self.assertEqual(observed['velocity_zup_mps'], [1.,0.,0.])
        self.assertEqual(receipt['lift_rpc']['raw_result'],True)
        self.assertEqual(receipt['zero_velocity_rpc']['raw_result'],True)
        self.assertTrue(client.paused)

    def test_cached_ground_collision_is_distinguished_from_new_contacts(self):
        old = dict(has_collided=True,time_stamp=90,object_name='Floor',object_id=0)
        self.assertFalse(new_execution_collision(old,old,100))
        self.assertFalse(new_execution_collision(dict(old,time_stamp=95),old,100))
        self.assertTrue(new_execution_collision(dict(old,time_stamp=101),old,100))
        self.assertTrue(new_execution_collision(dict(old,time_stamp=100,object_name='Obstacle'),old,100))
        self.assertFalse(new_execution_collision(dict(has_collided=False),old,100))
        with self.assertRaises(ValueError): new_execution_collision(dict(has_collided=True),old,100)


class GroupSelectionTests(unittest.TestCase):
    def test_explicit_choice_preserves_user_order_and_legacy_omission_retains_six(self):
        plans = six_plan_document()['plans']
        selected = select_plans(plans,['hermite_abom','bezier_ordinary'])
        self.assertEqual([p['group_id'] for p in selected], ['hermite_abom','bezier_ordinary'])
        with self.assertRaisesRegex(ValueError,'explicit'):
            select_plans(plans)
        self.assertEqual([p['group_id'] for p in plans], GROUP_IDS)

    def test_invalid_selection_fails_before_output_creation_or_airsim_client(self):
        for groups in ([],['unknown'],['bezier_ordinary','bezier_ordinary'],[''],['   ']):
            with self.subTest(groups=groups), TemporaryDirectory() as folder:
                source=Path(folder)/'airsim_replay_plan.json'
                write_json(source,six_plan_document())
                output=Path(folder)/'must_not_exist'
                constructor=mock.Mock(side_effect=AssertionError('selection must fail before client construction'))
                module=SimpleNamespace(MultirotorClient=constructor)
                args=SimpleNamespace(config=str(ROOT/'configs/curve_abom_airsim_pilot_20261029.json'),
                                     plans=str(source),groups=groups,output_dir=str(output),probe_only=False)
                with mock.patch.dict('sys.modules',{'airsim':module}), self.assertRaises(ValueError):
                    run(args)
                constructor.assert_not_called()
                self.assertFalse(output.exists())

    def test_runner_manifest_keeps_full_source_but_only_selected_execution_slots(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'airsim_replay_plan.json'
            document=six_plan_document()
            write_json(source,document)
            selected=['hermite_abom','bezier_ordinary']
            output=Path(folder)/'replay'
            c=config(); client=GroundClient(c['settings'])
            client.isApiControlEnabled=lambda **kwargs:False
            constructor=mock.Mock(return_value=client)
            def completed(_client,_lifecycle,plan,_config,_airsim,_output,**kwargs):
                return dict(group_id=plan['group_id'],plan=plan,status='COMPLETED',metrics={})
            args=SimpleNamespace(config=str(ROOT/'configs/curve_abom_airsim_pilot_20261029.json'),
                                 plans=str(source),groups=selected,output_dir=str(output),probe_only=False)
            with mock.patch.dict('sys.modules',{'airsim':SimpleNamespace(MultirotorClient=constructor)}), \
                 mock.patch('run_airsim_curve_replay_pilot.paused_ground_probe',return_value=dict(status='PASS')), \
                 mock.patch('run_airsim_curve_replay_pilot.replay_one',side_effect=completed) as replay, \
                 mock.patch('run_airsim_curve_replay_pilot.plots'), mock.patch('builtins.print'):
                self.assertEqual(run(args),0)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual([call.args[2]['group_id'] for call in replay.call_args_list],selected)
            self.assertEqual(manifest['selected_group_ids'],selected)
            self.assertEqual(manifest['unselected_group_ids'],[g for g in GROUP_IDS if g not in selected])
            self.assertEqual(manifest['plans_document'],document)
            self.assertEqual(manifest['planned_replays'],2)
            self.assertEqual(manifest['completed_replays'],2)
            self.assertEqual(manifest['not_run_group_ids'],[])
            frozen=next(s for s in manifest['sources'] if Path(s['path']).name=='airsim_replay_plan.json')
            self.assertEqual(json.loads((output/frozen['snapshot']).read_text(encoding='utf-8')),document)

    def test_full_audit_accepts_completed_subset_without_claiming_six_completed(self):
        with TemporaryDirectory() as folder:
            selected=['hermite_abom','bezier_ordinary']
            evidence,_=synthetic_audit_fixture(folder,selected)
            receipt=audit_directory(evidence,write=False)
            self.assertEqual(receipt['status'],'PASS')
            self.assertEqual(receipt['completion_scope'],'SELECTED_SUBSET')
            self.assertFalse(receipt['all_six_completed'])
            self.assertEqual(receipt['planned_replays'],2)
            self.assertEqual(receipt['completed_replays'],2)
            self.assertEqual(receipt['failed_replays'],0)
            self.assertEqual(receipt['selected_group_ids'],selected)
            self.assertEqual(receipt['not_run_group_ids'],[])
            self.assertEqual(receipt['original_selection_lineage']['plans'],6)

    def test_legacy_manifest_still_audits_all_six(self):
        with TemporaryDirectory() as folder:
            evidence,_=synthetic_audit_fixture(folder,GROUP_IDS,legacy=True)
            receipt=audit_directory(evidence,write=False)
            self.assertEqual(receipt['status'],'PASS')
            self.assertEqual(receipt['completion_scope'],'FULL_SIX')
            self.assertTrue(receipt['all_six_completed'])
            self.assertEqual(receipt['completed_replays'],6)

    def test_failed_selected_slot_does_not_count_unselected_groups_as_failed_or_not_run(self):
        with TemporaryDirectory() as folder:
            selected=['hermite_abom','bezier_ordinary']
            evidence,_=synthetic_audit_fixture(folder,selected,failed_first=True)
            receipt=audit_directory(evidence,write=False)
            self.assertEqual(receipt['status'],'PARTIAL')
            self.assertEqual(receipt['planned_replays'],2)
            self.assertEqual(receipt['attempted_replays'],1)
            self.assertEqual(receipt['failed_replays'],1)
            self.assertEqual(receipt['not_run_group_ids'],['bezier_ordinary'])
            self.assertEqual(len(receipt['unselected_group_ids']),4)

    def test_auditor_rejects_six_count_fabrication_for_a_single_selected_slot(self):
        with TemporaryDirectory() as folder:
            evidence,_=synthetic_audit_fixture(folder,['bezier_abom'])
            path=evidence/'manifest.json'
            manifest=json.loads(path.read_text(encoding='utf-8'))
            manifest['planned_replays']=6
            write_json(path,manifest)
            with self.assertRaisesRegex(AssertionError,'planned count'):
                audit_directory(evidence,write=False)

    def test_subset_audit_still_rejects_modified_unselected_model_source(self):
        with TemporaryDirectory() as folder:
            evidence,original=synthetic_audit_fixture(folder,['bezier_abom'])
            unselected=original/'groups/hermite_ordinary.json'
            unselected.write_text(unselected.read_text(encoding='utf-8')+' ',encoding='utf-8')
            with self.assertRaises(AssertionError): audit_directory(evidence,write=False)


if __name__ == '__main__':
    unittest.main()

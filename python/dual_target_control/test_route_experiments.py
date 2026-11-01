"""Two-policy geometry, waypoint decisions and explicit dispatch regressions."""
from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from .route_curve import route_spec,decode_route,route_metrics
from .curve_campaign import CurveCampaign
from route_control_experiments import plan_from_vector
from control_target_experiments import GROUPS


class RouteTests(unittest.TestCase):
    def plan(self,policy,representation='hermite'):
        spec=route_spec([[8,0,5],[12,6,7]],policy)
        return plan_from_vector([0,0,0,6,0,0,0,6],spec,representation)

    def test_equal_physical_basis_and_nonzero_pass_velocity(self):
        for policy in ['stop','pass']:
            h=decode_route(self.plan(policy));b=decode_route(self.plan(policy,'bezier'))
            t=np.linspace(0,h.duration_s,151)
            for d in range(3):np.testing.assert_allclose(h.sample(t,d),b.sample(t,d),atol=2e-10)
            speed=np.linalg.norm(h.sample(h.waypoint_times[0]['arrival_time_s'],1))
            self.assertAlmostEqual(speed,0 if policy=='stop' else 1.2,places=9)
            self.assertFalse(h.spec['future_optimization']['pass_speed_scale']['optimized'])

    def test_waypoint_rule_failure_is_not_hidden_by_terminal_success(self):
        for policy in ['stop','pass']:
            c=decode_route(self.plan(policy));t=np.unique(np.r_[np.linspace(0,c.duration_s,500),c.ends])
            states=[dict(reference_elapsed_s=float(x),position_zup_m=c.sample(x).tolist(),velocity_zup_mps=c.sample(x,1).tolist()) for x in t]
            commands=[dict(speed_clipped=False,slew_clipped=False) for _ in t[:-1]]
            self.assertTrue(route_metrics(c,states,commands)['control_error_gates_passed'])
            index=int(np.searchsorted(t,c.waypoint_times[0]['evaluation_time_s']))
            states[index]['velocity_zup_mps']=[.4,0,0] if policy=='stop' else [0,0,0]
            result=route_metrics(c,states,commands)
            self.assertFalse(result['control_error_gates_passed']);self.assertEqual(result['terminal_position_error_m'],0.)

    def test_reversal_and_degenerate_points_rejected(self):
        with self.assertRaises(ValueError):route_spec([[8,0,5],[0,0,5]],'pass')
        with self.assertRaises(ValueError):route_spec([[.1,0,5]],'stop')

    def test_independent_piecewise_metrics_and_waypoint_rules(self):
        from .curve_audit import independent_route_metrics
        for policy in ['stop','pass']:
            c=decode_route(self.plan(policy));times=np.unique(np.r_[np.linspace(0,c.duration_s,400),c.ends])
            states=[dict(reference_elapsed_s=float(t),position_zup_m=c.sample(t).tolist(),velocity_zup_mps=c.sample(t,1).tolist()) for t in times]
            commands=[dict(speed_clipped=False,slew_clipped=False) for _ in times[:-1]]
            producer=route_metrics(c,states,commands);auditor=independent_route_metrics(c,states,commands)
            self.assertAlmostEqual(producer['jerk_energy_m2_s5'],auditor['jerk_energy_m2_s5'],places=8)
            self.assertAlmostEqual(producer['quality_term'],auditor['quality_term'],places=8)
            self.assertTrue(auditor['control_error_checks']['waypoints'])


class FakeRuntime:
    def __init__(self):self.state=dict(status='READY',running=False,run_dir='',execution_id=None);self.calls=[]
    def snapshot(self):return deepcopy(self.state)
    def configure(self,cfg):self.config=cfg;self.calls.append(('configure',cfg))
    def start(self):self.calls.append(('start',None))


class CampaignTests(unittest.TestCase):
    def make(self,folder,selections=None):
        packages={}
        for policy in ['stop','pass']:
            path=Path(folder)/(policy+'.json');plans=[]
            for group in GROUPS:
                p=RouteTests().plan(policy,group.split('_')[0]);p['group_id']=group;plans.append(p)
            path.write_text(json.dumps(dict(plans=plans)),encoding='utf-8');packages[policy]=path
        cfg=dict(control={},airsim={},curve_experiment={})
        return CurveCampaign(cfg,packages,selections or [('stop','hermite_ordinary'),('pass','bezier_abom')],1,Path(folder)/'queue')

    def test_queue_waits_for_its_ack_and_dispatches_only_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();q.tick(r)
            self.assertEqual(q.index,0);self.assertEqual(r.config['curve_experiment']['plan']['group_id'],'hermite_ordinary')
            r.state.update(status='STOPPED',run_dir='stale')
            q.tick(r);self.assertEqual(q.index,0)
            r.state.update(execution_id=q.slots[0]['execution_id'],status='STOPPED',run_dir='new')
            q.tick(r);self.assertEqual(q.index,1);q.tick(r)
            self.assertEqual(r.config['curve_experiment']['plan']['group_id'],'bezier_abom')
            self.assertEqual(sum(c[0]=='start' for c in r.calls),2)

    def test_stop_before_dispatch_starts_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();q.stop_after_current();q.tick(r)
            self.assertFalse(q.active);self.assertEqual(r.calls,[])

    def test_configure_ack_cannot_reuse_previous_completed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();r.state.update(status='STOPPED',run_dir='previous')
            q.tick(r);r.state['execution_id']=q.slots[0]['execution_id'];q.tick(r)
            self.assertEqual(q.index,0);self.assertEqual(q.phase,'EXECUTING')

    def test_stop_after_current_does_not_launch_next(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();q.tick(r);q.stop_after_current()
            r.state.update(execution_id=q.slots[0]['execution_id'],status='STOPPED',run_dir='new')
            q.tick(r);q.tick(r)
            self.assertEqual(q.phase,'STOPPED');self.assertEqual(sum(c[0]=='start' for c in r.calls),1)

    def test_failed_landing_preserves_scene_and_halts_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();q.tick(r)
            r.state.update(execution_id=q.slots[0]['execution_id'],status='ERROR',running=True,run_dir='failed')
            q.tick(r);self.assertEqual(q.phase,'NEEDS_LANDING');self.assertFalse(q.active)

    def test_user_stop_is_retained_as_stop_instead_of_completed_flight(self):
        with tempfile.TemporaryDirectory() as folder:
            q=self.make(folder);r=FakeRuntime();q.tick(r)
            raw=Path(folder)/'raw';raw.mkdir();(raw/'summary.json').write_text(json.dumps(dict(outcome='USER_STOPPED')))
            r.state.update(execution_id=q.slots[0]['execution_id'],status='STOPPED',run_dir=str(raw))
            q.tick(r);self.assertEqual(q.slots[0]['status'],'USER_STOPPED')


class RoutePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from route_experiment_panel import RouteExperimentWindow,route_live_config
        from .route_live import RouteLiveRuntime
        from unittest.mock import Mock
        self.runtime=RouteLiveRuntime(route_live_config(),mode='demo');self.window=RouteExperimentWindow(self.runtime,self.runtime.config)
        self.runtime.start=Mock();self.runtime.configure=Mock()

    def tearDown(self):
        self.window.timer.stop();self.window.hide();self.runtime.shutdown();self.assertTrue(self.runtime.wait_closed(3));self.window.close()

    def test_switching_policy_method_and_loading_cached_route_do_not_fly(self):
        self.window.method.setCurrentIndex(5);self.window.policy.setCurrentIndex(1)
        self.window.prepare(['pass']);self.window.refresh()
        self.runtime.start.assert_not_called();self.assertEqual(self.runtime.snapshot()['status'],'IDLE')
        self.assertEqual(self.window.config['curve_experiment']['plan']['route']['policy'],'pass')

    def test_cancel_pending_batch_does_not_fly(self):
        self.window.pending_batch=dict(selections=[('stop','hermite_ordinary')],repeats=1)
        self.window.prepare(['stop']);self.window.stop_batch();self.window.refresh()
        self.runtime.start.assert_not_called();self.assertIsNone(self.window.campaign)

    def test_invalid_field_bounds_rejected_before_dispatch(self):
        self.window.inputs['x_min'].setValue(-4);self.window.inputs['x_max'].setValue(4)
        self.assertFalse(self.window._apply());self.runtime.start.assert_not_called();self.runtime.configure.assert_not_called()

    def test_batch_dialog_starts_with_no_methods_selected_and_cancel_does_not_fly(self):
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QCheckBox
        from curve_airsim_experiment_panel import TITLES
        observed=[]
        def inspect():
            dialog=self.app.activeModalWidget()
            try:observed.extend(b.isChecked() for b in dialog.findChildren(QCheckBox) if b.text() in TITLES.values())
            finally:dialog.reject()
        self.runtime._publish(connected=True,status='READY');QTimer.singleShot(0,inspect);self.window.batch_dialog()
        self.assertEqual(observed,[False]*6);self.runtime.start.assert_not_called();self.assertIsNone(self.window.pending_batch)

    def test_workspace_navigation_and_history_entry_do_not_fly(self):
        self.assertEqual(self.window.workspace.tabs.count(),3)
        for i in range(3):self.window.workspace.tabs.setCurrentIndex(i);self.app.processEvents()
        self.window.compare_history();self.assertEqual(self.window.workspace.library.filters['kind'].currentData(),'flight')
        self.runtime.start.assert_not_called()

    def test_data_operation_blocks_start_until_finished(self):
        self.window.workspace.library.busy=True
        try:
            self.window._start();self.window.refresh();self.runtime.start.assert_not_called();self.assertFalse(self.window.start_button.isEnabled())
        finally:self.window.workspace.library.busy=False


if __name__=='__main__':unittest.main()

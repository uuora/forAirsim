"""Preparation task, equal metre scales, selection and one-method budgeting."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prepared_route_planner import prepared_spec,generate_selected
from .route_curve import route_spec,decode_route
from route_control_experiments import plan_from_vector


class PreparedTaskTests(unittest.TestCase):
    def test_default_is_same_physical_task_as_protocol14(self):
        for policy in ('stop','pass'):
            points=[[8,0,5],[12,6,7],[6,10,6]]
            self.assertEqual(prepared_spec(points,policy),route_spec(points,policy))

    def test_arbitrary_start_preserves_both_bases_and_endpoint_states(self):
        for policy in ('stop','pass'):
            spec=prepared_spec([[0,0,5],[8,3,6]],policy,start=[-4,2,7])
            curves=[decode_route(plan_from_vector([.3,.1,-.2,6.]*2,spec,r)) for r in ('bezier','hermite')]
            t=np.linspace(0,curves[0].duration_s,137)
            for derivative in (0,1,2):np.testing.assert_allclose(curves[0].sample(t,derivative),curves[1].sample(t,derivative),atol=2e-9)
            np.testing.assert_allclose(curves[0].sample(0),[-4,2,7]);np.testing.assert_allclose(curves[0].sample(0,1),[0,0,0],atol=1e-10)

    def test_reject_bad_start_close_points_and_reversal(self):
        for start in ([float('nan'),0,5],[46,0,5],[0,0,2],[1,2]):
            with self.assertRaises(ValueError):prepared_spec([[8,0,5]],start=start)
        with self.assertRaises(ValueError):prepared_spec([[.5,0,5]],start=[0,0,5])
        with self.assertRaises(ValueError):prepared_spec([[8,0,5],[0,0,5]],'pass')

    def test_one_selected_method_does_not_generate_other_five(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'one';path=generate_selected(out,dict(task=dict(points=[[8,0,5]],start=[-2,1,5]),method='hermite_ordinary'))
            document=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual([p['group_id'] for p in document['plans']],['hermite_ordinary'])
            self.assertEqual(json.loads((out/'planning_summary.json').read_text())['full_surrogate_calls'],1)
            self.assertEqual(json.loads((out/'completion.json').read_text())['groups'],1)

    def test_selected_batch_preserves_preparation_and_each_policy(self):
        from prepared_route_panel import prepared_config,task_template
        from .curve_campaign import CurveCampaign
        from .test_route_experiments import FakeRuntime
        with tempfile.TemporaryDirectory() as folder:
            packages={}
            for policy in ('stop','pass'):
                spec=prepared_spec([[8,0,5],[12,6,7]],policy,start=[-3,1,5]);path=Path(folder)/(policy+'.json')
                path.write_text(json.dumps(dict(plans=[task_template(spec,g) for g in ('hermite_abom','bezier_ordinary')])))
                packages[policy]=path
            q=CurveCampaign(prepared_config(),packages,[('pass','hermite_abom'),('stop','bezier_ordinary')],1,Path(folder)/'batch')
            runtime=FakeRuntime();q.tick(runtime)
            self.assertEqual(runtime.config['preparation']['task']['start'],[-3.,1.,5.])
            self.assertEqual(runtime.config['preparation']['task']['policy'],'pass')
            self.assertEqual(runtime.config['preparation']['method'],'hermite_abom')
            self.assertTrue(runtime.config['curve_experiment']['plan']['planning_pending'])
            runtime.state.update(execution_id=q.slots[0]['execution_id'],status='STOPPED',run_dir='new')
            q.tick(runtime);q.tick(runtime)
            self.assertEqual(runtime.config['preparation']['task']['policy'],'stop')
            self.assertEqual(runtime.config['preparation']['method'],'bezier_ordinary')
            self.assertEqual(len(runtime.calls),4)

    def test_failure_before_a_plan_has_manifest_and_no_valid_score(self):
        from prepared_route_panel import prepared_config
        from .prepared_route_live import PreparedRouteRuntime
        from .curve_audit import audit_run
        import time
        with tempfile.TemporaryDirectory() as folder,patch('dual_target_control.runtime.ROOT',Path(folder)):
            runtime=PreparedRouteRuntime(prepared_config(),mode='demo')
            try:
                runtime.start();deadline=time.monotonic()+5
                while runtime.snapshot()['status']!='ERROR' and time.monotonic()<deadline:time.sleep(.01)
                run=Path(runtime.snapshot()['run_dir']);summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['outcome'],'FAILED');self.assertEqual(summary['airsim_reference_execution_calls'],0)
                self.assertFalse(summary['planning_after_arrival']);self.assertIsNone(summary['metrics'])
                self.assertEqual(audit_run(run)['status'],'NO_VALID_SCORE')
                self.assertTrue(json.loads((run/'live_manifest.json').read_text(encoding='utf-8'))['planning_pending'])
            finally:runtime.shutdown();runtime.wait_closed(5)


class ViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_equal_metre_scale_after_wide_and_tall_resize(self):
        from .ui import FlightView
        view=FlightView({})
        for size in ((920,430),(520,750)):
            view.resize(*size)
            for plane,directions in [('xy',[[1,0,0],[0,1,0]]),('xz',[[1,0,0],[0,0,1]])]:
                view.set_plane(plane);view._camera()
                lengths=np.linalg.norm(np.asarray(directions)@view.projection_matrix,axis=1)
                self.assertGreater(lengths[0],0);self.assertAlmostEqual(*lengths,places=10)
        view.close()

    def test_open_edit_and_switch_never_start_or_plan(self):
        from prepared_route_panel import PreparedRouteWindow,prepared_config
        from .prepared_route_live import PreparedRouteRuntime
        cfg=prepared_config();runtime=PreparedRouteRuntime(cfg)
        with patch('subprocess.Popen',side_effect=AssertionError('Opening/editing must not launch a worker')):
            window=PreparedRouteWindow(runtime,cfg)
            try:
                self.assertGreaterEqual(window.workspace.library.filters['protocol'].findData('15'),0)
                window.preparation_spins[0].setValue(-4);window.method.setCurrentIndex(5);self.assertTrue(window._apply())
                import time
                deadline=time.monotonic()+3
                while time.monotonic()<deadline and runtime.config['preparation']['task']['start'][0]!=-4:time.sleep(.01)
                self.assertEqual(runtime.config['preparation']['task']['start'],[-4.,0.,5.])
                self.assertFalse(runtime.snapshot()['running']);self.assertEqual(runtime.snapshot()['planned_path'],[])
                self.assertEqual(runtime.config['preparation']['method'],'hermite_abom_improved_candidate')
                runtime._publish(planned_path=[[0,0,5],[8,0,5]],trail=[[0,0,5]])
                window.method.setCurrentIndex(0)
                self.assertEqual(runtime.snapshot()['planned_path'],[]);self.assertEqual(runtime.snapshot()['trail'],[])
            finally:window.timer.stop();runtime.shutdown();runtime.wait_closed(5);window.deleteLater()


if __name__=='__main__':unittest.main()

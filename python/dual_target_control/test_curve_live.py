"""Protocol13 regressions without a simulator, native window or flight."""
import json,os,tempfile,time,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from dual_target_control.curve_live import LiveCurveRuntime
from dual_target_control.overview import marker_geometry
from dual_target_control.core import ControlConfig
from run_curve_live_panel import CurveLiveWindow,live_config
from PyQt5.QtWidgets import QApplication


class LiveCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_gui_selection_does_not_start_flight_and_start_uses_one_method(self):
        cfg=live_config();r=LiveCurveRuntime(cfg,mode='demo');w=CurveLiveWindow(r,cfg)
        r.start=Mock();r.configure=Mock()
        try:
            w.method.setCurrentIndex(5);self.app.processEvents()
            r.start.assert_not_called();self.assertIsNone(r._run_dir)
            w._start();r.start.assert_called_once()
            selected=r.configure.call_args.args[0]
            self.assertEqual(selected['curve_experiment']['plan']['group_id'],'hermite_abom_improved_candidate')
            self.assertEqual(selected['control']['max_speed'],6.)
            self.assertEqual(selected['curve_experiment']['time_scale'],1.25)
        finally:w.timer.stop();w.hide();self.close(r);w.close()

    def test_data_button_opens_current_run_instead_of_unknown_global_folder(self):
        cfg=live_config();r=LiveCurveRuntime(cfg,mode='demo');w=CurveLiveWindow(r,cfg)
        try:
            with tempfile.TemporaryDirectory() as folder,patch('run_curve_live_panel.QDesktopServices.openUrl') as opened:
                r._publish(run_dir=folder);w.open_data()
                self.assertEqual(Path(opened.call_args.args[0].toLocalFile()).resolve(),Path(folder).resolve())
        finally:w.timer.stop();w.hide();self.close(r);w.close()

    def close(self,r):
        r.mode='demo';r._active=False;r._flight_owned=False;r.shutdown()
        self.assertTrue(r.wait_closed(3))

    def test_opening_is_idle_and_does_not_launch_or_fly(self):
        with patch('dual_target_control.runtime.launch_simulator') as launch:
            r=LiveCurveRuntime(live_config(),mode='demo')
            try:
                time.sleep(.025);self.assertEqual(r.snapshot()['status'],'IDLE')
                self.assertIsNone(r._run_dir);launch.assert_not_called()
            finally:self.close(r)

    def test_single_goal_and_consumed_goal_marker_geometry(self):
        cfg=ControlConfig.from_dict(live_config()['control'])
        self.assertEqual(len(marker_geometry(dict(targets=[[8,0,5]]),cfg)['targets']),1)
        self.assertEqual(marker_geometry(dict(targets=[]),cfg)['targets'],[])

    def test_temporal_metric_uses_only_observed_command_intervals(self):
        cfg=live_config();r=LiveCurveRuntime(cfg,mode='demo')
        try:
            with tempfile.TemporaryDirectory() as folder:
                r._run_dir=Path(folder);r.live_curve=__import__('run_airsim_curve_replay_pilot').plan_curve(cfg['curve_experiment']['plan'])
                ts=np.linspace(0,r.live_curve.duration_s,100)
                for t in ts:
                    r.live_states.append(dict(sim_timestamp_ns=round(t*1e9),reference_elapsed_s=float(t),
                        position_zup_m=r.live_curve.sample(t).tolist(),velocity_zup_mps=r.live_curve.sample(t,derivative=1).tolist(),
                        reference_position_zup_m=r.live_curve.sample(t).tolist(),acceleration_zup_mps2=[0,0,0],
                        angular_velocity_body=[0,0,0],paused=False,yaw_rad=0))
                    r.live_commands.append(dict(reference_elapsed_s=float(t),speed_clipped=False,slew_clipped=False,limited=[0,0,0],yaw_command_deg=0))
                r._save('COMPLETED');result=json.loads((Path(folder)/'summary.json').read_text())
                self.assertIsNone(result['metric_error']);self.assertAlmostEqual(result['metrics']['position_rmse_m'],0)
                self.assertTrue(result['metrics']['control_error_gates_passed'])
        finally:self.close(r)

    def test_aborted_trial_cannot_claim_task_success_from_good_errors(self):
        cfg=live_config();r=LiveCurveRuntime(cfg,mode='demo')
        try:
            with tempfile.TemporaryDirectory() as folder:
                r._run_dir=Path(folder);r._compute_metrics=Mock(return_value=dict(control_error_gates_passed=True))
                r._save('USER_STOPPED');result=json.loads((Path(folder)/'summary.json').read_text())
                self.assertTrue(result['metrics']['control_error_gates_passed']);self.assertFalse(result['metrics']['task_success'])
                r._run_dir=None
        finally:self.close(r)


if __name__=='__main__':unittest.main()

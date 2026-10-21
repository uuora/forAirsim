import os,tempfile,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from t1_manual_workbench import ManualFlightWorkbench
from t1_obstacle_scene import ROOT

class ManualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):self.widget=ManualFlightWorkbench()
    def tearDown(self):self.widget.timer.stop();self.widget.close();self.app.processEvents()
    def test_flight_disabled_until_prepared(self):self.assertFalse(self.widget.fly.isEnabled());self.assertFalse(self.widget.stop.isEnabled())
    def test_building_mode_requires_scene_and_task(self):
        self.widget.mode.setCurrentIndex(1)
        with patch.object(self.widget,'start') as start:self.widget.prepare_plan();start.assert_not_called()
    def test_editing_instruction_invalidates_frozen_plan_and_task(self):
        self.widget.plan_path=Path('old');self.widget.task_path=Path('old');self.widget.instruction.setText('新指令')
        self.assertIsNone(self.widget.plan_path);self.assertIsNone(self.widget.task_path)
    def test_stop_signal_is_written_without_terminating_process(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'logs') as d:
            self.widget.busy=True;self.widget.action='run';self.widget.run_folder=Path(d)
            self.widget.request_stop();self.assertTrue((Path(d)/'manual_stop.request').is_file())
            self.assertFalse(self.widget.stop.isEnabled());self.widget.busy=False
    def test_frame_updates_live_map(self):
        import json
        self.widget.handle_line(json.dumps(dict(kind='LIVE_FRAME',acquisition=False,slot=2,gate_status='WAIT_NO_TARGET',position_ned_m=[.01,-3.,-3.],rgb_path='missing')))
        self.assertEqual(self.widget.map.trace[-1],[.01,-3.,-3.])
    def test_map_does_not_spawn_or_request_flight(self):
        with patch.object(self.widget,'start') as start:self.widget.refresh();start.assert_not_called()
        self.assertFalse(self.widget.busy)
if __name__=='__main__':unittest.main()

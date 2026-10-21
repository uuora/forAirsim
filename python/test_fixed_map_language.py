import copy, unittest
from unittest.mock import Mock,patch
from fixed_map_mission import OUT,read,sha
from fixed_map_language import validate_proposal,dispatch_proposal,interpret

class LanguageBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.task=read(OUT/'fixed_version/task.json')
        self.p=dict(disposition='ACCEPT',action='start',vehicle='DroneA',target='fixed_target',
            requested_height_m=None,requested_speed_limit_m_s=None,reason='开始固定任务')
    def test_new_coordinates_are_not_an_executable_extension(self):
        self.p['goal_xyz_m']=[0,0,5]
        with self.assertRaises(ValueError):validate_proposal(self.p,self.task)
    def test_frozen_parameter_changes_are_rejected(self):
        for key,value in [('requested_height_m',5),('requested_speed_limit_m_s',2),('requested_height_m',True)]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):validate_proposal(p,self.task)
    def test_drone_b_and_unknown_target_are_rejected(self):
        for key,value in [('vehicle','DroneB'),('target','other')]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):validate_proposal(p,self.task)
    def test_nonaccepted_action_is_not_dispatched(self):
        self.p.update(disposition='CLARIFY',action='none')
        e=Mock(task=self.task,stage='READY')
        with self.assertRaises(ValueError):dispatch_proposal(e,self.p,sha(OUT/'fixed_version/task.json'))
        e.submit_task.assert_not_called();e.command.assert_not_called()
    def test_start_uses_same_frozen_task_interface(self):
        e=Mock(task=self.task,stage='READY')
        dispatch_proposal(e,self.p,sha(OUT/'fixed_version/task.json'))
        e.submit_task.assert_called_once_with(self.task,'llm_preview_confirmed');e.command.assert_not_called()
    def test_stale_task_and_changed_phase_rejected_before_queue(self):
        for stage,digest in [('READY','old-hash'),('APPROACH',sha(OUT/'fixed_version/task.json'))]:
            e=Mock(task=self.task,stage=stage)
            with self.assertRaises(ValueError):dispatch_proposal(e,self.p,digest)
            e.submit_task.assert_not_called();e.command.assert_not_called()
    def test_hover_stop_uses_command_queue(self):
        self.p.update(action='stop',target='unspecified')
        e=Mock(task=self.task,stage='APPROACH')
        dispatch_proposal(e,self.p,sha(OUT/'fixed_version/task.json'))
        e.command.assert_called_once_with('stop','llm_preview_confirmed');e.submit_task.assert_not_called()
    def test_low_commit_headroom_rejects_before_loading_model(self):
        with patch('fixed_map_language.memory_headroom',return_value=dict(physical_available_bytes=8*2**30,commit_available_bytes=10**8)),patch('fixed_map_language.urllib.request.urlopen') as call:
            with self.assertRaisesRegex(ValueError,'余量不足'):interpret('开始',self.task)
            call.assert_not_called()

if __name__=='__main__':unittest.main()

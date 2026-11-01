"""A missing/failed flight must remain in the experiment denominator."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from analyze_control_full_flow import analyze


class AnalysisFailureTests(unittest.TestCase):
    def test_missing_and_not_run_trials_remain_in_both_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            tasks=[dict(id='fixed_missing',family='fixed',policy='stop',method='bezier_ordinary'),
                dict(id='online_not_run',family='online',arm='PGL',method='hermite_abom_improved_candidate',scene_seed=1401,optimizer_seed=21,config={'online':{'research':{'label':'test'}}})]
            (folder/'design_frozen.json').write_text(json.dumps({'tasks':tasks}),encoding='utf-8')
            digest=hashlib.sha256((folder/'design_frozen.json').read_bytes()).hexdigest()
            campaign=dict(status='ABORTED',design_sha256=digest,not_run=['online_not_run'],
                trials=[dict(task_id='fixed_missing',status='FAILED',closed_safely=True,run_dir=None,error='test failure')])
            (folder/'campaign.json').write_text(json.dumps(campaign),encoding='utf-8')
            with patch('analyze_control_full_flow.plots',return_value=[]):output=analyze(folder)
            result=json.loads((output/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['fixed']),1);self.assertEqual(len(result['online']),1)
            self.assertFalse(result['fixed'][0]['task_success']);self.assertFalse(result['online'][0]['task_success'])
            self.assertIsNone(result['online'][0]['targets']);self.assertIsNone(result['fixed'][0]['position_rmse_m'])
            self.assertEqual(result['not_run'],['online_not_run']);self.assertEqual(len(result['issues']),2)


if __name__=='__main__':unittest.main()

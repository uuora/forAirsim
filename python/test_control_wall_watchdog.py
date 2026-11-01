import json,sys,time,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from run_control_wall_watchdog_research import enforce,read_incumbents


class WatchdogTests(unittest.TestCase):
    def write(self,path,events):path.write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')

    def test_unpublished_and_late_candidates_cannot_replace_incumbent(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'events.jsonl';events=[]
            for i,finish,score in ((1,.2,.5),(2,.4,.3),(3,1.1,.1)):
                events.extend([dict(kind='BEGIN',index=i,parameters=[i]),dict(kind='COMPLETE',record=dict(index=i,parameters=[i],score=score,feasible=True,finish_s=finish))])
                if i!=2:events.append(dict(kind='PUBLISHED',index=i,ready_s=finish+.01))
            self.write(p,events);receipt=read_incumbents(p,1.)
            self.assertEqual(receipt['best']['index'],1)
            self.assertEqual(receipt['completed_calls'],3)

    def test_truncated_last_line_is_retained_as_unfinished(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'events.jsonl';self.write(p,[dict(kind='BEGIN',index=1,parameters=[0])])
            with p.open('a') as f:f.write('{"kind":"COMPLETE"')
            receipt=read_incumbents(p,1.)
            self.assertTrue(receipt['incomplete_trailing_line'])
            self.assertEqual(receipt['started_without_complete'],1)
            self.assertIsNone(receipt['best'])

    def test_watchdog_cancels_only_owned_slow_worker_and_returns_prefix(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'events.jsonl';self.write(p,[])
            started=time.perf_counter()
            receipt=enforce([sys.executable,'-c','import time;time.sleep(5)'],started=started,budget=.15,journal=p,log=Path(folder)/'worker.log')
            self.assertTrue(receipt['process_cancelled'])
            self.assertIsNone(receipt['primary_best_feasible_score'])
            self.assertLess(receipt['host_return_wall_s'],2.)


if __name__=='__main__':unittest.main()

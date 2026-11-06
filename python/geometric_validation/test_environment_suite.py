"""Suite scheduling stubs are not flight evidence."""
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .environment_suite import run_suite, collect_suite, render_suite_page, _write_csv
from .planners import METHODS


class SuiteTests(unittest.TestCase):
    def test_summary_opens_with_3d_image_and_switchable_named_environments(self):
        class Tags(HTMLParser):
            def __init__(self):super().__init__();self.items=[]
            def handle_starttag(self,tag,attrs):self.items.append((tag,dict(attrs)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);requests=[];campaigns=[]
            for index in range(2):
                folder=root/('E'+str(index));folder.mkdir()
                (folder/'trajectory_3d.png').write_bytes(b'derived-image-fixture')
                requests.append(dict(environment_id=str(index),label='E'+str(index)))
                campaigns.append(dict(environment=str(index),path=str(folder)))
            (root/'suite.json').write_text(json.dumps(dict(requests=requests)),encoding='utf-8')
            summary=dict(results=[],environments=campaigns,actual_flights=6,requested=6,accepted=6)
            (root/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
            page=render_suite_page(root);parser=Tags();parser.feed(page.read_text(encoding='utf-8'))
            tabs=[a for _,a in parser.items if a.get('role')=='tab']
            panels=[a for _,a in parser.items if a.get('role')=='tabpanel']
            images=[a['src'] for tag,a in parser.items if tag=='img']
            self.assertEqual(len(tabs),2)
            self.assertEqual(tabs[0]['aria-selected'],'true')
            self.assertEqual(tabs[0]['aria-controls'],panels[0]['id'])
            self.assertNotIn('hidden',panels[0]);self.assertIn('hidden',panels[1])
            self.assertEqual(images,['E0/trajectory_3d.png','E1/trajectory_3d.png'])
            self.assertTrue((root/'page_provenance.json').is_file())

    def test_collection_is_repeatable_and_reaudit_failure_rejects_report_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);flight=root/'flight';flight.mkdir();campaign=root/'campaign';campaign.mkdir()
            def save(path,value):path.write_text(json.dumps(value),encoding='utf-8')
            cfg=dict(cases={m+'_pass':dict(segment_durations=[4,4]) for m in METHODS},
                     comparison=dict(task=dict(points_ned=[[0,0,-5],[4,0,-5],[8,0,-5]])))
            save(root/'config.json',cfg);save(root/'design_frozen.json',{})
            original=dict(control_constraints_passed=True,data_audit_passed=True,waypoint_passage_passed=True)
            report=dict(original,accepted=True,normal_completion=True,
                        cleanup=dict(landed=True,disarmed=True,api_control_released=True))
            save(flight/'audit.json',original);save(flight/'report.json',report)
            save(flight/'manifest.json',dict(comparison_fingerprint='fixture'))
            sample=dict(timestamp_ns=1,phase='TRACKING',t=0,position=[0,0,-5],
                        velocity=[0,0,0],acceleration=[1,2,3],quaternion_wxyz=[1,0,0,0])
            (flight/'samples.jsonl').write_text(json.dumps(sample)+'\n',encoding='utf-8')
            save(campaign/'campaign.json',dict(results=[dict(method='bezier',status='RECORDED',
                                                           run_dir=str(flight),report=report)]))
            save(root/'suite.json',dict(requests=[dict(environment_id='fixture',label='fixture',
                 campaign_dir=str(campaign),config=str(root/'config.json'),status='RECORDED')]))
            failed=dict(original,data_audit_passed=False,issues=['fixture discrepancy'])
            with patch('geometric_validation.audit.audit',return_value=failed):
                first=collect_suite(root)
                second=collect_suite(root)
            self.assertEqual(first,second)
            self.assertEqual(first['actual_flights'],1)
            self.assertEqual(first['accepted'],0)
            self.assertEqual(first['not_run'],2)
            self.assertFalse(first['reports_match_reaudit'])
            self.assertEqual(json.loads((flight/'audit_initial.json').read_text(encoding='utf-8')),original)
            self.assertIn('acceleration_x',(flight/'states_export.csv').read_text(encoding='utf-8-sig'))

    def test_changed_derived_csv_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'export.csv';_write_csv(path,[dict(value=1)])
            before=path.read_bytes()
            with self.assertRaises(ValueError):_write_csv(path,[dict(value=2)])
            self.assertEqual(path.read_bytes(),before)

    def test_unsafe_cleanup_halts_later_environments_and_preserves_all_requests(self):
        calls=[]
        def fake(config,output,methods,launch,runner=None):
            calls.append(config)
            return Path(output),[dict(method=m,safe_cleanup=False,accepted=False) for m in METHODS]
        with tempfile.TemporaryDirectory() as tmp:
            folder=run_suite(Path(tmp)/'suite',campaign_runner=fake)
            data=json.loads((folder/'suite.json').read_text(encoding='utf-8'))
            self.assertEqual(len(calls),1)
            self.assertEqual(data['design']['requested_flights'],15)
            self.assertEqual([r['status'] for r in data['requests']],['RECORDED']+['NOT_RUN']*4)

    def test_control_failures_do_not_prevent_independent_tasks_after_safe_cleanup(self):
        calls=[]
        def fake(config,output,methods,launch,runner=None):
            calls.append(config)
            return Path(output),[dict(method=m,safe_cleanup=True,accepted=False) for m in METHODS]
        with tempfile.TemporaryDirectory() as tmp:
            folder=run_suite(Path(tmp)/'suite',campaign_runner=fake)
            data=json.loads((folder/'suite.json').read_text(encoding='utf-8'))
            self.assertEqual(len(calls),5)
            self.assertTrue(all(r['status']=='RECORDED' for r in data['requests']))

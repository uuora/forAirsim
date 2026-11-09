"""Synthetic selected-record tests, never flight evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .campaign import PLANNER_CONFIG
from .planners import common_fingerprint
from .record_comparison import discover_records, generate_comparison, group_records, inspect_record


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def fixture(root,method='bezier',name=None,cfg=None,failed=False):
    folder=Path(root)/(name or 'geometric_synthetic_'+method);folder.mkdir(parents=True)
    cfg=deepcopy(cfg or json.loads(PLANNER_CONFIG.read_text(encoding='utf-8')))
    cfg['synthetic_test_fixture']=True;save(folder/'config.json',cfg)
    source=folder/'source'/'synthetic.txt';source.parent.mkdir();source.write_text('Synthetic source snapshot',encoding='utf-8')
    manifest=dict(protocol=cfg['protocol'],case=method+'_pass',created_utc_ns=1793952086912846900,
        config_sha256=hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(),
        sources={'synthetic.txt':hashlib.sha256(source.read_bytes()).hexdigest()},
        local_preflight=dict(files={'synthetic.dll':dict(sha256='synthetic')}),
        live=dict(loaded_settings={'synthetic_test_fixture':True}))
    manifest['comparison_fingerprint']=hashlib.sha256(json.dumps(dict(common=common_fingerprint(cfg),
        sources=manifest['sources'],runtime=manifest['local_preflight']['files'],settings=manifest['live']['loaded_settings']),sort_keys=True).encode()).hexdigest()
    save(folder/'manifest.json',manifest)
    cleanup=dict(landed=True,disarmed=True,api_control_released=True)
    report=dict(normal_completion=not failed,control_constraints_passed=not failed,data_audit_passed=True,
        waypoint_passage_passed=True,accepted=not failed,actual_completion_time_s=None if failed else 12.,cleanup=cleanup)
    save(folder/'report.json',report);save(folder/'cleanup.json',cleanup);save(folder/'events.json',[])
    save(folder/'audit.json',dict(original_audit='preserve these bytes'))
    rows=[dict(phase='TRACKING',t=t,position=[t,0,-5],velocity=[.8,0,0],
        reference=dict(p=[t,0,-5]),attitude_reference_error_rad=0.) for t in (.1,.2)]
    (folder/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return folder


def fake_audit(directory):
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    result={k:report[k] for k in ('control_constraints_passed','data_audit_passed','waypoint_passage_passed')}
    result.update(issues=[],metrics=dict(position_error_m=dict(peak=0.),attitude_reference_error_rad=dict(peak=0.)))
    save(directory/'audit.json',result)
    return result


class RecordComparisonTests(unittest.TestCase):
    def test_discovery_includes_gui_campaign_failures_and_incomplete_records_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=fixture(root/'logs',failed=True)
            second=fixture(root/'reports/geometric_three_planners_synthetic/flights','hermite')
            partial=root/'logs/geometric_incomplete';partial.mkdir()
            records=discover_records(root,[first])
            self.assertEqual(len(records),3)
            self.assertEqual(sum(r['eligible'] for r in records),2)
            self.assertFalse(next(r for r in records if r['path']==str(first))['normal_completion'])
            self.assertTrue(inspect_record(second)['eligible'])

    def test_different_controller_source_or_runtime_remains_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=fixture(tmp)
            cfg=json.loads(PLANNER_CONFIG.read_text(encoding='utf-8'));cfg['limits']['position_error_m']+=.1
            second=fixture(tmp,'hermite',cfg=cfg)
            self.assertEqual(len(group_records([inspect_record(first),inspect_record(second)])),2)
            manifest=json.loads((first/'manifest.json').read_text(encoding='utf-8'))
            manifest['sources']['synthetic.txt']='changed';save(first/'manifest.json',manifest)
            self.assertFalse(inspect_record(first)['eligible'])

    def test_duplicate_method_rejected_but_partial_method_selection_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=inspect_record(fixture(tmp));second=inspect_record(fixture(tmp,name='geometric_repeat'))
            self.assertEqual(len(group_records([first])),1)
            with self.assertRaisesRegex(ValueError,'每种方案'):group_records([first,second])
            with self.assertRaisesRegex(ValueError,'重复选择'):group_records([first,first])
            with self.assertRaises(ValueError):group_records([])

    def test_changed_config_and_unfinished_reports_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=fixture(tmp)
            save(folder/'report.json',{'normal_completion':None})
            self.assertFalse(inspect_record(folder)['eligible'])
            (folder/'config.json').write_text('{}',encoding='utf-8')
            self.assertFalse(inspect_record(folder)['eligible'])

    def test_selected_output_preserves_all_original_bytes_and_failed_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);flights=[fixture(root,m,failed=m=='hermite') for m in ('bezier','hermite','bspline')]
            before={p:p.read_bytes() for flight in flights for p in flight.rglob('*') if p.is_file()}
            with patch('geometric_validation.audit.audit',side_effect=fake_audit), \
                 patch('socket.create_connection',side_effect=AssertionError('Unexpected RPC')), \
                 patch('subprocess.Popen',side_effect=AssertionError('Unexpected launch')):
                page=generate_comparison(flights,root/'derived')
            summary=json.loads((page.parent/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['new_flights'],0)
            rows=summary['groups'][0]['results'];self.assertEqual([r['accepted'] for r in rows],[True,False,True])
            self.assertIsNone(rows[1]['actual_time_s'])
            copied=Path(rows[0]['run_dir'])
            self.assertEqual((copied/'audit_original.json').read_bytes(),(flights[0]/'audit.json').read_bytes())
            self.assertNotEqual((copied/'audit.json').read_bytes(),(flights[0]/'audit.json').read_bytes())
            for path,data in before.items():self.assertEqual(path.read_bytes(),data)
            self.assertTrue((page.parent/'trajectory_view.html').is_file())
            self.assertTrue(json.loads((page.parent/'provenance.json').read_text(encoding='utf-8'))['original_inputs_unchanged'])
            with self.assertRaises(FileExistsError):generate_comparison(flights,root/'derived')
            with self.assertRaisesRegex(ValueError,'原始记录'):generate_comparison(flights,flights[0]/'derived')

    def test_tampered_source_is_rejected_without_writing_to_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=fixture(tmp);(folder/'source/synthetic.txt').write_text('tampered',encoding='utf-8')
            before=(folder/'audit.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'哈希'):generate_comparison([folder],Path(tmp)/'derived')
            self.assertEqual((folder/'audit.json').read_bytes(),before)
            self.assertTrue((Path(tmp)/'derived/generation_error.json').is_file())

    def test_new_lookup_finds_selected_group_for_matching_environment_only(self):
        from .environments import matching_comparison, config_for
        cfg=json.loads(PLANNER_CONFIG.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            group=Path(tmp)/'geometric_selected_comparison_synthetic/group_1';group.mkdir(parents=True)
            save(group/'config_frozen.json',cfg);page=group/'index.html';page.write_text('synthetic',encoding='utf-8')
            self.assertEqual(matching_comparison(cfg,tmp),page)
            self.assertIsNone(matching_comparison(json.loads(config_for('planar_s').read_text(encoding='utf-8')),tmp))

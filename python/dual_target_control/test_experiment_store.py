"""Data retention, stale previews, archive round trips and UI filtering."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from .experiment_store import ExperimentStore,checked

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store=ExperimentStore(self.root)
    def tearDown(self):self.temp.cleanup()
    def view(self,name='old'):
        path=self.root/'reports'/name;path.mkdir(parents=True)
        (path/'index.html').write_text('saved chart',encoding='utf-8');(path/'trajectory.png').write_bytes(b'original pixels')
        write(path/'provenance.json',dict(condition_fingerprint='fixed',input_sha256={'raw':'a'*64}))
        return path
    def flight(self,outcome='COMPLETED',passed=True):
        path=self.root/'logs'/'run';write(path/'live_manifest.json',dict(protocol_revision=14,method='hermite_ordinary',plan=dict(seed=1,route=dict(policy='stop')),condition_fingerprint='fixed',mode='airsim'))
        write(path/'summary.json',dict(outcome=outcome,metrics=dict(control_error_gates_passed=passed)))
        (path/'states.csv').write_text('raw evidence');return path
    def test_failed_and_incomplete_raw_runs_are_retained_and_not_cleanup_candidates(self):
        raw=self.flight('FAILED');before=(raw/'states.csv').read_bytes();rows=self.store.scan();self.assertEqual(rows[0]['status'],'FAILED')
        with self.assertRaises(ValueError):self.store.preview_archive(['logs/run'])
        self.assertEqual((raw/'states.csv').read_bytes(),before)
        (raw/'summary.json').unlink();self.assertEqual(self.store.scan()[0]['status'],'INCOMPLETE')
    def test_annotations_and_pins_survive_rescan_and_new_store_instance(self):
        self.view();self.store.scan();self.store.annotate('reports/old',tags='开发集',note='保留负结果',pinned=True)
        store=ExperimentStore(self.root);store.scan();row=store.records()[0];self.assertEqual(row['note'],'保留负结果')
        with self.assertRaises(ValueError):store.preview_archive(['reports/old'])
    def test_archive_restore_round_trip_keeps_every_byte(self):
        path=self.view();before={p.name:p.read_bytes() for p in path.iterdir()};preview=self.store.preview_archive(['reports/old']);receipt=self.store.archive(preview)[0]
        self.assertFalse(path.exists());self.assertTrue(any(r['status']=='ARCHIVED' for r in self.store.records()))
        restored=self.store.restore(receipt['id']);self.assertEqual({p.name:p.read_bytes() for p in restored.iterdir()},before)
    def test_stale_preview_rejected_before_any_original_is_removed(self):
        path=self.view();preview=self.store.preview_archive(['reports/old']);(path/'trajectory.png').write_bytes(b'new pixels')
        with self.assertRaises(ValueError):self.store.archive(preview)
        self.assertEqual((path/'index.html').read_text(),'saved chart')
    def test_unknown_or_raw_file_in_derived_view_prevents_cleanup(self):
        path=self.view();(path/'states.csv').write_text('raw')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/old'])
        (path/'states.csv').unlink();(path/'important.txt').write_text('unknown')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/old'])
    def test_active_view_and_default_protected_views_are_not_archived(self):
        path=self.view()
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/old'],active=[path])
        default=self.view('route_dashboard_v2')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/route_dashboard_v2'])
    def test_restore_never_overwrites_new_content(self):
        path=self.view();receipt=self.store.archive(self.store.preview_archive(['reports/old']))[0]
        path.mkdir();(path/'index.html').write_text('new work')
        with self.assertRaises(ValueError):self.store.restore(receipt['id'])
        self.assertEqual((path/'index.html').read_text(),'new work')
    def test_path_escape_and_archive_id_escape_are_rejected(self):
        with self.assertRaises(ValueError):checked(self.root,'../outside')
        with self.assertRaises(ValueError):self.store.restore('../archive')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/../../elsewhere'])
    def test_archive_corruption_is_detected_before_restore(self):
        path=self.view();receipt=self.store.archive(self.store.preview_archive(['reports/old']))[0]
        manifest=self.store.home/'archive'/(receipt['id']+'.json');item=json.loads(manifest.read_text(encoding='utf-8'));item['files'][0]['sha256']='0'*64;write(manifest,item)
        with self.assertRaises(ValueError):self.store.restore(receipt['id'])
        self.assertFalse(path.exists())
    def test_same_raw_input_views_grouped_without_claiming_byte_duplicates(self):
        self.view('one');self.view('two');rows=self.store.scan();self.assertEqual([r['same_input_views'] for r in rows],[2,2])
    def test_new_run_can_be_indexed_without_a_full_rescan(self):
        path=self.flight();self.store.index_paths([path]);self.assertEqual(self.store.records()[0]['status'],'PASS')
    def test_frozen_planning_package_never_qualifies_as_cleanup(self):
        path=self.root/'reports'/'plan';write(path/'airsim_replay_plan.json',dict(plans=[]));write(path/'completion.json',dict(status='COMPLETE'))
        self.assertEqual(self.store.scan()[0]['kind'],'plan')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/plan'])
    def test_valid_json_with_wrong_metadata_type_does_not_disappear(self):
        path=self.flight();write(path/'summary.json',[]);self.assertEqual(self.store.scan()[0]['status'],'INCOMPLETE')
    def test_source_changed_run_is_not_classified_as_valid_pass(self):
        path=self.flight();write(path/'summary.json',dict(outcome='COMPLETED',sources_unchanged=False,metrics=dict(control_error_gates_passed=True)))
        self.assertEqual(self.store.scan()[0]['status'],'INVALID')
    def test_malformed_nested_metadata_is_visible_instead_of_dropping_trial(self):
        path=self.flight();write(path/'live_manifest.json',dict(plan=['invalid']))
        row=self.store.scan()[0];self.assertEqual(row['status'],'INCOMPLETE');self.assertIn('元数据结构异常',row['error'])
    def test_archived_view_keeps_protocol_policy_and_notes_for_filtering(self):
        raw=self.flight();view=self.view();write(view/'provenance.json',dict(condition_fingerprint='fixed',input_sha256={str(raw/'live_manifest.json'):'hash'}))
        self.store.annotate('reports/old',tags='旧图');self.store.archive(self.store.preview_archive(['reports/old']))
        archived=next(r for r in self.store.records() if r.get('archive_id'))
        self.assertEqual(archived['protocol'],14);self.assertEqual(archived['policy'],'stop');self.assertEqual(archived['tags'],'旧图')
    def test_current_workspace_keeps_new_failed_runs_but_separates_development_suites(self):
        raw=self.flight('FAILED');self.assertEqual(self.store.scan()[0]['collection'],'current')
        write(raw/'summary.json',dict(outcome='FAILED',config=dict(curve_experiment=dict(batch_id='route_protocol14_pilot_v1'))))
        self.assertEqual(self.store.scan()[0]['collection'],'history')
    def test_history_visibility_is_reversible_persistent_and_never_moves_raw_data(self):
        raw=self.flight();original=(raw/'states.csv').read_bytes();self.store.scan()
        self.store.set_visibility(['logs/run'],'history');self.assertEqual(ExperimentStore(self.root).scan()[0]['collection'],'history')
        self.assertEqual((raw/'states.csv').read_bytes(),original)
        self.store.set_visibility(['logs/run'],'current');self.assertEqual(self.store.records()[0]['collection'],'current')
    def test_only_latest_same_input_view_is_current(self):
        raw=self.flight()
        for name in ['control_view_one','control_view_two']:
            path=self.view(name);write(path/'provenance.json',dict(input_sha256={str(raw/'live_manifest.json'):'hash'},condition_fingerprint='fixed'))
        rows=self.store.scan();views=[r for r in rows if r['kind']=='view']
        self.assertEqual(sum(r['collection']=='current' for r in views),1)
        self.assertEqual(sum(r['collection']=='history' for r in views),1)
    def test_symlink_is_rejected(self):
        target=self.view()
        try:(self.root/'reports'/'link').symlink_to(target,target_is_directory=True)
        except OSError:self.skipTest('Creating symlinks is unavailable')
        with self.assertRaises(ValueError):self.store.preview_archive(['reports/link'])

if __name__=='__main__':unittest.main()

"""Reusable data UI flows, without launching or moving a simulated drone."""
import os,json,tempfile,time,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from .experiment_library_ui import DataLibrary
from .experiment_store import ExperimentStore

class LibraryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store=ExperimentStore(self.root)
        self.owner=SimpleNamespace(locked=lambda:False,runtime=SimpleNamespace(snapshot=lambda:{}),last_results=None,workspace=SimpleNamespace(show_results=lambda _:None))
        self.library=None
    def tearDown(self):
        if self.library:self.library.close();self.library.deleteLater();self.app.processEvents()
        self.temp.cleanup()
    def run_record(self,name,policy):
        path=self.root/'logs'/name;path.mkdir(parents=True)
        (path/'live_manifest.json').write_text(json.dumps(dict(protocol_revision=14,method='hermite_ordinary',plan=dict(seed=1,route=dict(policy=policy)),condition_fingerprint=policy,mode='airsim')))
        (path/'summary.json').write_text(json.dumps(dict(outcome='COMPLETED',metrics=dict(control_error_gates_passed=True))))
    def make(self):self.library=DataLibrary(self.owner,self.root);return self.library
    def test_filter_and_checked_records_survive_filter_changes(self):
        self.run_record('one','stop');self.run_record('two','pass');self.store.scan();lib=self.make()
        lib.filters['policy'].setCurrentIndex(lib.filters['policy'].findData('stop'));self.assertEqual(lib.proxy.rowCount(),1)
        lib.check_visible();self.assertEqual(lib.selected()[0]['policy'],'stop')
        lib.filters['policy'].setCurrentIndex(0);lib.search.setText('Hermite');self.assertEqual(lib.proxy.rowCount(),2)
        self.assertEqual(len(lib.selected()),1);lib.clear_checked();self.assertEqual(lib.selected(),[])
    def test_busy_flight_blocks_scanning_and_cleanup_jobs(self):
        lib=self.make();self.owner.locked=lambda:True;lib.scan();self.assertFalse(lib.busy);self.assertIn('结束后',lib.message.text())
    def test_history_is_collapsed_by_default_and_protocol13_expands_it(self):
        self.run_record('new','stop');self.run_record('old','single')
        path=self.root/'logs/old/live_manifest.json';old=json.loads(path.read_text());old['protocol_revision']=13;path.write_text(json.dumps(old))
        self.store.scan();lib=self.make();self.assertEqual(lib.proxy.rowCount(),1)
        lib.filters['protocol'].setCurrentIndex(lib.filters['protocol'].findData('13'))
        self.assertTrue(lib.history.isChecked());self.assertEqual(lib.proxy.rowCount(),1)
        lib.history.setChecked(False);self.assertEqual(lib.filters['protocol'].currentData(),'');self.assertEqual(lib.proxy.rowCount(),1)
    def test_restore_selected_archive_through_ui(self):
        view=self.root/'reports'/'old';view.mkdir(parents=True);(view/'index.html').write_text('saved')
        (view/'provenance.json').write_text(json.dumps(dict(input_sha256={'raw':'a'},condition_fingerprint='x')))
        self.store.archive(self.store.preview_archive(['reports/old']));lib=self.make();archive=next(r for r in lib.model.rows if r.get('archive_id'))
        lib.model.checked.add(archive['id']);lib.restore();deadline=time.monotonic()+5
        while lib.busy and time.monotonic()<deadline:self.app.processEvents();time.sleep(.01)
        self.assertFalse(lib.busy);self.assertTrue((view/'index.html').is_file());self.assertIn('恢复',lib.message.text())

if __name__=='__main__':unittest.main()

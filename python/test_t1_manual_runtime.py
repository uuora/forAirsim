"""Read-only prerequisites and identity-bound close; zero real simulator calls."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import t1_manual_runtime as runtime

class ManualRuntimeTests(unittest.TestCase):
    def test_snapshot_does_not_silently_copy_modified_source(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=runtime.ROOT/'logs') as d:
            root=Path(d);source=root/'source.py';source.write_text('original',encoding='utf-8');sha=hashlib.sha256(source.read_bytes()).hexdigest()
            plan_path=root/'plan.json';plan_path.write_text('{}',encoding='utf-8');source.write_text('modified',encoding='utf-8')
            plan=dict(source_inventory=[dict(path=str(source),relative_path='python/source.py',sha256=sha)],legacy_source_inventory=[])
            with self.assertRaises(ValueError):runtime.snapshot_manual_run(plan_path,plan,root/'run')
    def test_engine_pid_reuse_is_rejected(self):
        identity=dict(pid=123,executable='owned.exe',creation_time=1.)
        with patch.object(runtime,'owner_identity',return_value=dict(identity,creation_time=2.)):
            with self.assertRaisesRegex(RuntimeError,'IDENTITY'):runtime.assert_owner(dict(engine=identity))
    def test_building_flight_requires_ready_scene_before_freezing(self):
        with self.assertRaises(ValueError):runtime.prepare_manual('zero',runtime.DEFAULT_SCENE)
    def test_occupied_baseline_is_rejected_before_plan_creation(self):
        with patch.object(runtime,'rpc_busy',return_value=True):
            with self.assertRaisesRegex(ValueError,'已有场景'):runtime.prepare_manual('zero')

if __name__=='__main__':unittest.main()

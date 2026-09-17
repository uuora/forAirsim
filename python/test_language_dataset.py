import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from language_dataset import DatasetStore, content_hash


ALIGN = dict(disposition='ACCEPT', routine='visual_alignment', vehicle='DroneA', reason='OK')
RANGE = dict(disposition='ACCEPT', routine='visual_range', vehicle='DroneA', reason='OK')


class LanguageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.store = DatasetStore(self.folder)

    def exported(self):
        manifest = self.store.export_training(self.folder / 'exports')
        payload = Path(manifest['output_path']).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest['sha256'])
        return manifest, [json.loads(line) for line in payload.decode('utf-8').splitlines()]

    def test_pending_capture_is_never_implicitly_approved(self):
        row = self.store.collect_candidate('观察气球', ALIGN, source_metadata={'model': 'candidate-only'})
        self.assertEqual(row['review_status'], 'pending_human_review')
        self.assertEqual(row, self.store.rows()[0])
        self.assertIsNone(self.store.review_for(row))
        manifest, records = self.exported()
        self.assertEqual(records, [])
        self.assertEqual(manifest['status'], 'empty_no_approved_development_rows')
        self.assertEqual(manifest['exclusion_counts'], {'pending_human_review': 1})
        self.assertFalse(self.store.seed_path.exists())

    def test_edited_label_export_preserves_prompt_and_provenance(self):
        row = self.store.collect_candidate('靠近气球', ALIGN, scenario_group='range_family')
        event = self.store.review(row, 'approved', 'human_01', RANGE, '纠正接近任务')
        manifest, records = self.exported()
        self.assertEqual(records[0]['messages'][:2], row['messages'][:2])
        self.assertEqual(json.loads(records[0]['messages'][2]['content']), RANGE)
        for field in ('id', 'scenario_group', 'source', 'split', 'content_hash'):
            self.assertEqual(records[0][field], row[field])
        self.assertEqual(manifest['included'][0]['review_event_id'], event['event_id'])
        self.assertEqual(self.store.rows()[0]['messages'][2], row['messages'][2])

    def test_latest_decision_overrides_prior_approval_without_deleting_it(self):
        row = self.store.collect_candidate('观察气球')
        first = self.store.review(row, 'approved', 'human_01', ALIGN)
        last = self.store.review(row, 'rejected', 'human_02', None, '语义需要重新核查')
        self.assertEqual(self.store.review_for(row), last)
        manifest, records = self.exported()
        self.assertFalse(records)
        self.assertEqual(manifest['exclusion_counts'], {'rejected': 1})
        events = self.store.review_path.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(events), 2)
        self.assertEqual(json.loads(events[0]), first)

    def test_smoke_eval_never_exports_even_if_approved(self):
        smoke = self.store.collect_candidate('保持观察', ALIGN, split='smoke_eval', scenario_group='heldout')
        self.store.review(smoke, 'approved', 'human_01', ALIGN)
        duplicate = self.store.collect_candidate('保持观察', ALIGN, scenario_group='different_name')
        self.store.review(duplicate, 'approved', 'human_01', ALIGN)
        valid = self.store.collect_candidate('靠近观察', RANGE)
        self.store.review(valid, 'approved', 'human_01', RANGE)
        manifest, records = self.exported()
        self.assertEqual([row['id'] for row in records], [valid['id']])
        self.assertEqual(manifest['exclusion_counts'], {'reserved_split': 1, 'reserved_instruction_overlap': 1})
        with self.assertRaisesRegex(ValueError, 'cross dataset splits'):
            self.store.collect_candidate('其他改写', ALIGN, scenario_group='heldout')

    def test_changed_source_content_invalidates_old_review(self):
        original = self.store.collect_candidate('观察气球', ALIGN)
        self.store.review(original, 'approved', 'human_01', ALIGN)
        # Simulate external editing; the normal API never edits stored candidates.
        changed = dict(original)
        changed.pop('content_hash')
        changed['source'] = 'external_edit'
        self.store.candidate_path.write_text(json.dumps(changed, ensure_ascii=False) + '\n', encoding='utf-8')
        current = self.store.rows()[0]
        self.assertNotEqual(content_hash(original), content_hash(current))
        self.assertIsNone(self.store.review_for(current))
        manifest, records = self.exported()
        self.assertFalse(records)
        self.assertEqual(manifest['exclusion_counts'], {'stale_review': 1})
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.review(original, 'approved', 'human_01', ALIGN)

    def test_invalid_review_or_label_leaves_no_review_events(self):
        row = self.store.collect_candidate('观察气球')
        for decision, reviewer, plan in [('approved', ' ', ALIGN), ('yes', 'human_01', ALIGN),
                                         ('approved', 'human_01', None), ('approved', 'human_01', {}),
                                         ('rejected', 'human_01', dict(ALIGN, routine='shell'))]:
            with self.assertRaises(ValueError):
                self.store.review(row, decision, reviewer, plan)
        self.assertFalse(self.store.review_path.exists())
        with self.assertRaises(ValueError):
            self.store.collect_candidate('观察气球', dict(ALIGN, vehicle='DroneB'))
        self.assertEqual(len(self.store.rows()), 1)

    def test_group_conflict_from_external_seed_is_excluded(self):
        row = self.store.collect_candidate('开发表达', ALIGN, scenario_group='family')
        self.store.review(row, 'approved', 'human_01', ALIGN)
        seed = dict(row, id='heldout_id', split='test')
        seed.pop('content_hash')
        seed['messages'] = row['messages'][:]
        seed['messages'][1] = dict(role='user', content='保留测试表达')
        self.store.seed_path.write_text(json.dumps(seed, ensure_ascii=False), encoding='utf-8')
        manifest, records = self.exported()
        self.assertFalse(records)
        self.assertEqual(manifest['exclusion_counts'], {'reserved_split': 1, 'scenario_group_leakage': 1})

    def test_seed_is_immutable_and_exports_are_unique_snapshots(self):
        row = self.store.collect_candidate('观察气球', ALIGN)
        seed = dict(row, id='seed_01')
        seed.pop('content_hash')
        original_bytes = json.dumps(seed, ensure_ascii=False).encode('utf-8')
        self.store.seed_path.write_bytes(original_bytes)
        seed_row = next(record for record in self.store.rows() if record['id'] == 'seed_01')
        self.store.review(seed_row, 'approved', 'human_01', ALIGN)
        first, _ = self.exported()
        second, _ = self.exported()
        self.assertEqual(self.store.seed_path.read_bytes(), original_bytes)
        self.assertNotEqual(first['output_path'], second['output_path'])
        self.assertEqual(first['sha256'], second['sha256'])

    def test_malformed_append_log_fails_visibly(self):
        self.store.candidate_path.write_text('{"id":', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'malformed JSON'):
            self.store.collect_candidate('观察气球', ALIGN)
        self.assertEqual(self.store.candidate_path.read_text(encoding='utf-8'), '{"id":')


if __name__ == '__main__':
    unittest.main()

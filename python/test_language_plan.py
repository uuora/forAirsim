import unittest
import json
from pathlib import Path
from language_plan import parse_plan, preview_steps, validate_plan

class LanguagePlanTests(unittest.TestCase):
    def test_valid_preview_has_no_contact(self):
        plan = dict(disposition='ACCEPT', routine='visual_range', vehicle='DroneA', reason='OK')
        self.assertEqual(preview_steps(plan)[-1], 'land_both')
        self.assertNotIn('contact', preview_steps(plan))

    def test_rejects_unknown_action_and_extra_code(self):
        for changes in [dict(routine='shell'), dict(code='print(1)'), dict(vehicle='DroneB')]:
            plan = dict(disposition='ACCEPT', routine='visual_range', vehicle='DroneA', reason='OK')
            plan.update(changes)
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_rejects_inconsistent_disposition(self):
        with self.assertRaises(ValueError):
            validate_plan(dict(disposition='UNSUPPORTED', routine='visual_range', vehicle='DroneA', reason='OK'))

    def test_no_code_or_fenced_json_repair(self):
        for text in ['print(1)', '```json\n{}\n```', '{} {}']:
            with self.assertRaises(ValueError):
                parse_plan(text)

    def test_clarification_has_no_executable_preview(self):
        self.assertEqual(preview_steps(dict(disposition='CLARIFY', routine='none', vehicle='none',
                                           reason='MISSING_INFORMATION')), [])

    def test_seed_groups_do_not_cross_splits_and_labels_validate(self):
        path = Path(__file__).resolve().parents[1] / 'datasets/language_tasks_v0/seed.jsonl'
        groups = {}
        identifiers = set()
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            self.assertNotIn(row['id'], identifiers)
            identifiers.add(row['id'])
            previous = groups.setdefault(row['scenario_group'], row['split'])
            self.assertEqual(previous, row['split'])
            self.assertEqual(row['review_status'], 'pending_human_review')
            parse_plan(row['messages'][-1]['content'])

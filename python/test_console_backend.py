"""Console boundary tests: no simulator, subprocess, model, or flight calls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import console_backend


ALIGN = dict(disposition='ACCEPT', routine='visual_alignment', vehicle='DroneA', reason='OK')
RANGE = dict(disposition='ACCEPT', routine='visual_range', vehicle='DroneA', reason='OK')
INSTRUCTION = '让A观察中心红气球，然后返回降落。'


def record(plan=None):
    return dict(instruction=INSTRUCTION, schema_valid=True, plan=deepcopy(ALIGN if plan is None else plan))


def ground_snapshot():
    return dict(vehicles={
        'DroneA': dict(position=[0.0, 0.0, 0.8], speed=0.0, api_control=False),
        'DroneB': dict(position=[0.0, 4.0, 0.8], speed=0.01, api_control=False),
    })


class ConsoleBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.python = self.root / '.venv/python.exe'
        root_patch = patch.object(console_backend, 'ROOT', self.root)
        python_patch = patch.object(console_backend, 'PYTHON', self.python)
        root_patch.start()
        python_patch.start()
        self.addCleanup(root_patch.stop)
        self.addCleanup(python_patch.stop)
        # If a boundary test accidentally reaches external work, fail immediately.
        for name in ('local_server', 'infer', 'sim_client'):
            guard = patch.object(console_backend, name, side_effect=AssertionError('Unexpected external call: ' + name))
            guard.start()
            self.addCleanup(guard.stop)
        process_guard = patch.object(console_backend.subprocess, 'Popen',
                                     side_effect=AssertionError('Unexpected subprocess'))
        process_guard.start()
        self.addCleanup(process_guard.stop)

    def test_allowed_plans_map_only_to_fixed_observation_script(self):
        for plan, suffix in ((ALIGN, []), (RANGE, ['--range'])):
            with self.subTest(routine=plan['routine']):
                approved = console_backend.snapshot_id(INSTRUCTION, plan)
                command = console_backend.execution_command(INSTRUCTION, record(plan), approved)
                self.assertEqual(command, [str(self.python), '-u',
                                           str(self.root / 'python/run_visual_alignment.py')] + suffix)
                self.assertNotIn('--contact', command)
                self.assertNotIn(INSTRUCTION, command)

    def test_text_or_plan_edit_invalidates_previous_snapshot(self):
        approved = console_backend.snapshot_id(INSTRUCTION, ALIGN)
        with self.assertRaises(ValueError):
            console_backend.execution_command('靠近中心气球', record(), approved)
        with self.assertRaises(ValueError):
            console_backend.execution_command(INSTRUCTION, record(RANGE), approved)
        changed_record = record()
        changed_record['instruction'] = '旧指令'
        with self.assertRaises(ValueError):
            console_backend.execution_command(INSTRUCTION, changed_record, approved)
        for missing_or_wrong in (None, '', 'a_different_hash'):
            with self.subTest(snapshot=missing_or_wrong), self.assertRaises(ValueError):
                console_backend.execution_command(INSTRUCTION, record(), missing_or_wrong)

    def test_unvalidated_or_nonexecutable_plan_cannot_execute(self):
        approved = console_backend.snapshot_id(INSTRUCTION, ALIGN)
        unvalidated = record()
        unvalidated['schema_valid'] = False
        missing_validation = record()
        missing_validation.pop('schema_valid')
        missing_plan = record()
        missing_plan.pop('plan')
        for item in (unvalidated, missing_validation, missing_plan):
            with self.subTest(record=item), self.assertRaises(ValueError):
                console_backend.execution_command(INSTRUCTION, item, approved)
        for disposition, reason in (('CLARIFY', 'MISSING_INFORMATION'),
                                    ('UNSUPPORTED', 'CAPABILITY_UNAVAILABLE')):
            plan = dict(disposition=disposition, routine='none', vehicle='none', reason=reason)
            with self.subTest(disposition=disposition), self.assertRaises(ValueError):
                console_backend.execution_command(INSTRUCTION, record(plan),
                                                  console_backend.snapshot_id(INSTRUCTION, plan))

    def test_extra_fields_or_unsupported_actions_cannot_be_dispatched(self):
        approved = console_backend.snapshot_id(INSTRUCTION, ALIGN)
        for edits in ({'code': 'print(1)'}, {'routine': 'contact'}, {'vehicle': 'DroneB'},
                      {'routine': 'python/run_balloon_mission.py'}):
            item = record(dict(ALIGN, **edits))
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                console_backend.execution_command(INSTRUCTION, item, approved)

    def test_known_ground_and_stationary_vehicles_are_ready(self):
        self.assertIsNone(console_backend.require_ground_ready(ground_snapshot()))

    def test_ground_gate_rejects_motion_api_control_and_airborne_states(self):
        invalid_items = [dict(speed=0.08), dict(speed=0.2), dict(api_control=True),
                         dict(position=[0.0, 0.0, -2.0]), dict(position=[0.0, 0.0, 0.65]),
                         dict(position=[0.0, 0.0, 1.1])]
        for vehicle in ('DroneA', 'DroneB'):
            for edits in invalid_items:
                snapshot = ground_snapshot()
                snapshot['vehicles'][vehicle].update(edits)
                with self.subTest(vehicle=vehicle, edits=edits), self.assertRaises(ValueError):
                    console_backend.require_ground_ready(snapshot)

    def test_ground_gate_rejects_nonfinite_position_and_speed(self):
        for vehicle in ('DroneA', 'DroneB'):
            for value in (float('nan'), float('inf'), float('-inf')):
                for field in ('x', 'y', 'z', 'speed'):
                    snapshot = ground_snapshot()
                    item = snapshot['vehicles'][vehicle]
                    if field == 'speed':
                        item['speed'] = value
                    else:
                        item['position']['xyz'.index(field)] = value
                    with self.subTest(vehicle=vehicle, value=value, field=field), self.assertRaises(ValueError):
                        console_backend.require_ground_ready(snapshot)

    def test_cancelled_request_epoch_stays_rejected_until_new_request(self):
        backend = console_backend.ConsoleBackend(lambda message: None)
        self.assertTrue(backend.folder.is_relative_to(self.root))
        old_epoch = backend.request_epoch()
        self.assertIsInstance(old_epoch, int)
        backend.cancel_pending()
        new_epoch = backend.request_epoch()
        self.assertGreater(new_epoch, old_epoch)
        self.assertTrue(backend._stop_requested)
        # Retrying a queued, cancelled request must never clear the stop flag.
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                backend._begin(request_epoch=old_epoch)
            self.assertTrue(backend._stop_requested)
        backend._begin(request_epoch=new_epoch)
        self.assertFalse(backend._stop_requested)
        self.assertEqual(backend.request_epoch(), new_epoch)


if __name__ == '__main__':
    unittest.main()

"""An old DLL or a stale startup log must not authorize the new physics profile."""
from datetime import datetime
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .dynamics import read_live_dynamics, validate_receipt


class DynamicsReceiptTests(unittest.TestCase):
    def setUp(self):
        self.profile = dict(Schema=1, Name='dual_target_v1', MassKg=1., ThrustToWeightRatio=2.5)
        self.receipt = dict(schema=1, profile='dual_target_v1', vehicle='DroneControl', model='GenericQuadX',
            mass_kg=1., rotor_count=4, max_thrust_per_rotor_n=6.12915625,
            max_torque_per_rotor_nm=.08148191745, total_max_thrust_n=24.516625,
            nominal_twr=2.5, nominal_gravity_mps2=9.80665,
            inertia_diag_kgm2=[.0067211875, .0080406875, .014278875], max_rpm=7746.3072,
            nominal_hover_throttle=.4, min_velocity_throttle=.3)

    def test_matching_initialized_physics_is_verified(self):
        result = validate_receipt(self.profile, self.profile, self.receipt, 'DroneControl')
        self.assertTrue(result['verified'])
        self.assertAlmostEqual(result['total_max_thrust_n'] / (result['mass_kg'] * 9.80665), 2.5)

    def test_changed_settings_invalid_physics_or_hover_floor_is_rejected(self):
        for changes in ({'vehicle': 'Other'}, {'mass_kg': 2.}, {'nominal_twr': 1.7047},
                        {'max_thrust_per_rotor_n': 4.179446}, {'min_velocity_throttle': .5},
                        {'inertia_diag_kgm2': [1, -1, 1]}, {'mass_kg': True}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                validate_receipt(self.profile, self.profile, dict(self.receipt, **changes), 'DroneControl')
        with self.assertRaises(RuntimeError):
            validate_receipt(self.profile, dict(self.profile, ThrustToWeightRatio=1.7), self.receipt, 'DroneControl')

    def test_old_plugin_and_old_log_cannot_pass_using_only_new_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / 'logs/dual_target_control_ue.log'
            log.parent.mkdir()
            opened = datetime(2026, 10, 15, 19, 0, 0)
            settings = {'Vehicles': {'DroneControl': {'DynamicsProfile': self.profile}}}
            config = {'airsim': {'settings': str(root / 'settings.json'), 'vehicle': 'DroneControl'}}
            (root / 'settings.json').write_text(json.dumps(settings), encoding='utf8')
            client = SimpleNamespace(getSettingsString=lambda: json.dumps(settings))
            process = Mock()
            process.cmdline.return_value = ['UE4Editor.exe', '-abslog=' + str(log)]
            process.create_time.return_value = opened.timestamp()
            header = 'Log file open, ' + opened.strftime('%m/%d/%y %H:%M:%S') + '\n'
            with patch('dual_target_control.dynamics.ROOT', root), \
                 patch('dual_target_control.native_window.Win32GameWindow._process_pid', return_value=123), \
                 patch('psutil.Process', return_value=process):
                log.write_text(header + 'LogInit: Old plugin without physics receipt\n', encoding='utf8')
                with self.assertRaisesRegex(RuntimeError, '未提供动力回执'):
                    read_live_dynamics(client, config)
                log.write_text(header + 'LogTemp: Display: DTC_DYNAMICS '
                               + json.dumps(self.receipt) + '\n', encoding='utf8')
                result = read_live_dynamics(client, config)
                self.assertTrue(result['verified'])
                self.assertEqual(hashlib.sha256(Path(result['log_snapshot']).read_bytes()).hexdigest(),
                                 result['log_sha256'])
                process.create_time.return_value += 120
                with self.assertRaisesRegex(RuntimeError, '旧进程'):
                    read_live_dynamics(client, config)


if __name__ == '__main__':
    unittest.main()

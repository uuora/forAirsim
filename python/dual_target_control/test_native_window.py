"""Process/window identity regressions; all desktop and process APIs are mocked."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from dual_target_control.native_window import (
    MF_DISABLED, MF_GRAYED, SC_CLOSE, Win32GameWindow, _Win32Api)
from dual_target_control.simulator import ROOT


class FakeWindows:
    def __init__(self):
        self.rows = [(777, 200, 'UnrealWindow'), (444, 400, 'UnrealWindow'),
                     (888, 200, 'ConsoleWindowClass')]
        self.visible = False
        self.actions = []
        self.owners = {}
        self.tools = set()
        self.titles = {777: 'Blocks (64-bit, PCD3D_SM5)', 444: 'Other UE Scene'}

    def windows(self):
        return list(self.rows)

    def is_window(self, hwnd):
        return any(row[0] == hwnd for row in self.rows)

    def window_pid(self, hwnd):
        return next((row[1] for row in self.rows if row[0] == hwnd), 0)

    def window_class(self, hwnd):
        return next((row[2] for row in self.rows if row[0] == hwnd), '')

    def window_owner(self, hwnd):
        return self.owners.get(hwnd, 0)

    def is_tool_window(self, hwnd):
        return hwnd in self.tools

    def window_title(self, hwnd):
        return self.titles.get(hwnd, '')

    def is_visible(self, hwnd):
        return self.visible

    def disable_close(self, hwnd):
        self.actions.append(('disable_close', hwnd))

    def resize(self, hwnd, width, height):
        self.actions.append(('resize', hwnd, width, height))

    def show(self, hwnd):
        self.actions.append(('show', hwnd))
        self.visible = True

    def hide(self, hwnd):
        self.actions.append(('hide', hwnd))
        self.visible = False

    def foreground(self, hwnd):
        self.actions.append(('foreground', hwnd))
        return False  # Windows focus denial must not invalidate a successful show.


def listener(pid=200, port=41460, ip='0.0.0.0', status=psutil.CONN_LISTEN):
    return SimpleNamespace(pid=pid, laddr=SimpleNamespace(ip=ip, port=port), status=status)


class NativeWindowTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(airsim=dict(host='127.0.0.1', port=41460,
                           editor='D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe',
                           project='external/AirSim/Unreal/Environments/Blocks/Blocks.uproject',
                           map='/Game/DroneControlOverview/Maps/DroneControlOverview',
                           settings='configs/dual_target_airsim_settings.json'))
        self.arguments = [self.config['airsim']['editor'],
                          str((ROOT / self.config['airsim']['project']).resolve()),
                          self.config['airsim']['map'], '-game', '-windowed',
                          '-settings=' + str((ROOT / self.config['airsim']['settings']).resolve())]
        self.process = SimpleNamespace(exe=lambda: self.arguments[0],
            name=lambda: 'UE4Editor.exe', cmdline=lambda: list(self.arguments))
        self.api = FakeWindows()
        self.network = Mock(return_value=[listener(), listener(pid=400, port=41458)])
        self.process_lookup = Mock(return_value=self.process)
        self.start_patch('dual_target_control.native_window.sys.platform', 'win32')
        self.start_patch('dual_target_control.native_window.psutil.net_connections', self.network)
        self.start_patch('dual_target_control.native_window.psutil.Process', self.process_lookup)
        self.api_factory = self.start_patch('dual_target_control.native_window._Win32Api',
                                            return_value=self.api)

    def start_patch(self, *args, **kwargs):
        patcher = patch(*args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_show_hide_and_visibility_only_use_the_verified_game_hwnd(self):
        window = Win32GameWindow(self.config, preferred_pid=200)
        self.assertFalse(window.is_visible())
        self.assertEqual(window.show(), {'pid': 200, 'hwnd': 777})
        self.assertTrue(window.is_visible())
        self.assertEqual(window.hide(), {'pid': 200, 'hwnd': 777})
        self.assertFalse(window.is_visible())
        self.assertEqual(self.api.actions, [('disable_close', 777), ('resize', 777, 1200, 800),
                         ('show', 777), ('foreground', 777), ('hide', 777)])
        self.assertTrue(all(row[1] == 777 for row in self.api.actions))
        self.assertTrue(all(row.args == (200,) for row in self.process_lookup.call_args_list))
        self.assertTrue(all(row.kwargs == {'kind': 'tcp'} for row in self.network.call_args_list))

    def test_hide_does_not_close_or_terminate_any_process(self):
        window = Win32GameWindow(self.config)
        window.hide()
        self.assertEqual(self.api.actions, [('hide', 777)])
        self.assertFalse(hasattr(self.process, 'terminate'))
        self.assertFalse(hasattr(self.api, 'close'))

    def test_other_project_map_settings_editor_and_non_game_are_rejected(self):
        original = list(self.arguments)
        mutations = [(1, 'D:/Other/Other.uproject'),
                     (2, '/Game/DroneControlEmpty/Maps/DroneControlEmpty'),
                     (5, '-settings=D:/Other/settings.json'),
                     (0, 'D:/Other/UE4Editor.exe'), (3, '-editor'),
                     (1, 'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject')]
        for index, value in mutations:
            with self.subTest(index=index, value=value):
                self.arguments = list(original)
                self.arguments[index] = value
                with self.assertRaises(RuntimeError):
                    Win32GameWindow(self.config).show()
                self.assertEqual(self.api.actions, [])
        self.arguments = original + ['-settings=D:/Other/settings.json']
        with self.assertRaises(RuntimeError):
            Win32GameWindow(self.config).hide()
        self.assertEqual(self.api.actions, [])

    def test_wrong_preferred_pid_has_no_authority_to_select_another_ue(self):
        with self.assertRaisesRegex(RuntimeError, '不一致'):
            Win32GameWindow(self.config, preferred_pid=400).show()
        self.process_lookup.assert_not_called()
        self.assertEqual(self.api.actions, [])

    def test_missing_or_ambiguous_tcp_port_owner_is_rejected(self):
        for listeners in ([], [listener(pid=None)], [listener(), listener(pid=400)],
                          [listener(status='ESTABLISHED')], [listener(ip='127.0.0.9')]):
            with self.subTest(listeners=listeners):
                self.network.return_value = listeners
                with self.assertRaisesRegex(RuntimeError, '唯一'):
                    Win32GameWindow(self.config).show()
                self.assertEqual(self.api.actions, [])

    def test_missing_or_ambiguous_unreal_windows_are_rejected(self):
        self.api.titles[999] = 'Second titled game viewport'
        for rows in ([], [(444, 400, 'UnrealWindow')],
                     [(777, 200, 'UnrealWindow'), (999, 200, 'UnrealWindow')]):
            with self.subTest(rows=rows):
                self.api.rows = rows
                with self.assertRaisesRegex(RuntimeError, '唯一'):
                    Win32GameWindow(self.config).show()
                self.assertEqual(self.api.actions, [])

    def test_non_windows_and_remote_hosts_never_enumerate_desktop(self):
        with patch('dual_target_control.native_window.sys.platform', 'linux'):
            window = Win32GameWindow(self.config)
            with self.assertRaisesRegex(RuntimeError, 'Windows'):
                window.show()
            self.assertFalse(window.is_visible())
        remote = deepcopy(self.config)
        remote['airsim']['host'] = '192.168.1.10'
        with self.assertRaisesRegex(RuntimeError, '远端'):
            Win32GameWindow(remote).hide()
        self.network.assert_not_called()
        self.api_factory.assert_not_called()

    def test_hide_rechecks_port_identity_after_a_previous_show(self):
        window = Win32GameWindow(self.config)
        window.show()
        self.api.actions.clear()
        self.arguments[2] = '/Game/Other/Maps/Other'
        with self.assertRaisesRegex(RuntimeError, '不一致'):
            window.hide()
        self.assertEqual(self.api.actions, [])
        self.assertFalse(window.is_visible())

    def test_lost_hwnd_identity_is_rejected_before_mutation(self):
        self.api.window_pid = lambda hwnd: 400
        with self.assertRaisesRegex(RuntimeError, '唯一'):
            Win32GameWindow(self.config).show()
        self.assertEqual(self.api.actions, [])

    def test_main_game_window_is_selected_with_unreal_screenshot_notifications(self):
        self.api.rows.extend([(1001, 200, 'UnrealWindow'), (1002, 200, 'UnrealWindow'),
                              (1003, 200, 'UnrealWindow')])
        self.api.owners[1001] = 777
        self.api.tools.add(1002)
        self.api.titles.update({1001: 'Screenshot saved', 1002: 'UE notification'})
        window = Win32GameWindow(self.config)
        self.assertEqual(window.show(), {'pid': 200, 'hwnd': 777})
        self.assertEqual(window.hide(), {'pid': 200, 'hwnd': 777})
        self.assertTrue(all(action[1] == 777 for action in self.api.actions))

    def test_notifications_without_a_main_game_window_are_rejected(self):
        self.api.rows = [(1001, 200, 'UnrealWindow'), (1002, 200, 'UnrealWindow'),
                         (1003, 200, 'UnrealWindow')]
        self.api.owners[1001] = 777
        self.api.tools.add(1002)
        self.api.titles.update({1001: 'Screenshot saved', 1002: 'UE notification'})
        window = Win32GameWindow(self.config)
        with self.assertRaisesRegex(RuntimeError, '唯一'):
            window.show()
        with self.assertRaisesRegex(RuntimeError, '唯一'):
            window.hide()
        self.assertFalse(window.is_visible())
        self.assertEqual(self.api.actions, [])

    def test_main_window_identity_is_rechecked_before_each_mutation(self):
        window = Win32GameWindow(self.config)
        original_resize = self.api.resize

        def become_tool_window(hwnd, width, height):
            original_resize(hwnd, width, height)
            self.api.tools.add(hwnd)

        self.api.resize = become_tool_window
        with self.assertRaisesRegex(RuntimeError, '失效'):
            window.show()
        self.assertNotIn(('show', 777), self.api.actions)

    def test_system_close_disabling_can_be_explicitly_turned_off(self):
        self.config['native_window'] = dict(disable_close=False)
        Win32GameWindow(self.config).show()
        self.assertEqual(self.api.actions, [('resize', 777, 1200, 800),
                                           ('show', 777), ('foreground', 777)])

    def test_second_gui_show_reuses_already_disabled_close_command(self):
        menu_state = {'value': 0}
        user32 = SimpleNamespace(GetSystemMenu=Mock(return_value=123),
            GetMenuState=Mock(side_effect=lambda *args: menu_state['value']),
            EnableMenuItem=Mock(), DrawMenuBar=Mock())

        def enable(*args):
            previous = menu_state['value']
            menu_state['value'] = MF_GRAYED
            return previous

        user32.EnableMenuItem.side_effect = enable
        native_api = object.__new__(_Win32Api)
        native_api.user32 = user32
        self.api.disable_close = native_api.disable_close
        first = Win32GameWindow(self.config)
        self.assertEqual(first.show(), {'pid': 200, 'hwnd': 777})
        first.hide()
        second = Win32GameWindow(self.config)
        self.assertEqual(second.show(), {'pid': 200, 'hwnd': 777})
        user32.EnableMenuItem.assert_called_once_with(123, SC_CLOSE, MF_GRAYED)
        user32.DrawMenuBar.assert_called_once_with(777)
        self.assertEqual([action for action in self.api.actions if action[0] == 'show'],
                         [('show', 777), ('show', 777)])

    def test_unavailable_or_disabled_close_is_already_safe_to_show(self):
        for state in (0xFFFFFFFF, MF_GRAYED, MF_DISABLED, MF_GRAYED | MF_DISABLED):
            with self.subTest(state=state):
                native_api = object.__new__(_Win32Api)
                native_api.user32 = SimpleNamespace(GetSystemMenu=Mock(return_value=123),
                    GetMenuState=Mock(return_value=state), EnableMenuItem=Mock(), DrawMenuBar=Mock())
                self.assertTrue(native_api.disable_close(777))
                native_api.user32.EnableMenuItem.assert_not_called()
                native_api.user32.DrawMenuBar.assert_not_called()


if __name__ == '__main__':
    unittest.main()

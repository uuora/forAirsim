"""Offscreen record-selection workflows; no simulation or flight is started."""
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from .record_comparison import inspect_record
from .record_comparison_ui import RecordComparisonDialog
from .test_record_comparison import fixture


class SelectionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def wait(self,condition):
        deadline=time.monotonic()+5
        while not condition() and time.monotonic()<deadline:QTest.qWait(10)
        self.assertTrue(condition())

    def test_open_filter_select_generate_and_reopen_do_not_call_rpc(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[fixture(tmp,m) for m in ('bezier','hermite','bspline')]
            records=[inspect_record(p) for p in paths];threads=[]
            def generate(selected):threads.append(threading.get_ident());return Path(tmp)/'derived/index.html'
            with patch('socket.create_connection',side_effect=AssertionError('RPC')), \
                 patch('subprocess.Popen',side_effect=AssertionError('Launch')), \
                 patch('PyQt5.QtGui.QDesktopServices.openUrl',return_value=True) as opened:
                dialog=RecordComparisonDialog(root=tmp,scanner=lambda *a:records,generator=generate)
                dialog.show();self.wait(lambda:not dialog.busy)
                self.assertFalse(dialog.generate_button.isEnabled());self.assertEqual(dialog.table.rowCount(),3)
                for row in range(3):dialog.table.item(row,0).setCheckState(Qt.Checked)
                self.assertTrue(dialog.generate_button.isEnabled())
                dialog.search.setText('hermite');self.assertEqual(dialog.table.rowCount(),1)
                self.assertEqual(len(dialog.checked),3)
                dialog.search.clear();dialog.generate();self.wait(lambda:not dialog.busy)
                self.assertNotEqual(threads,[threading.get_ident()]);opened.assert_called_once()
                self.assertTrue(dialog.open_button.isEnabled());self.assertTrue(dialog.last_page)
                dialog.clear_selection();self.assertFalse(dialog.generate_button.isEnabled());dialog.close()

    def test_duplicate_method_disables_generation_and_failed_record_stays_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[inspect_record(fixture(tmp,failed=True)),inspect_record(fixture(tmp,name='geometric_repeat'))]
            dialog=RecordComparisonDialog(root=tmp,scanner=lambda *a:records);dialog.show();self.wait(lambda:not dialog.busy)
            dialog.table.item(0,0).setCheckState(Qt.Checked);self.assertTrue(dialog.generate_button.isEnabled())
            dialog.table.item(1,0).setCheckState(Qt.Checked);self.assertFalse(dialog.generate_button.isEnabled())
            self.assertIn('每种方案',dialog.status.text());dialog.close()

    def test_close_waits_for_background_work_and_does_not_open_a_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=inspect_record(fixture(tmp));gate=threading.Event()
            def generate(*_):gate.wait(3);return Path(tmp)/'derived/index.html'
            dialog=RecordComparisonDialog(root=tmp,scanner=lambda *a:[record],generator=generate)
            dialog.show();self.wait(lambda:not dialog.busy);dialog.table.item(0,0).setCheckState(Qt.Checked)
            with patch('PyQt5.QtGui.QDesktopServices.openUrl') as opened:
                try:
                    dialog.generate();dialog.close();self.assertTrue(dialog.isVisible());self.assertTrue(dialog.busy)
                finally:gate.set()
                self.wait(lambda:not dialog.busy);self.assertFalse(dialog.isVisible());opened.assert_not_called()

    def test_open_generation_failure_preserves_existing_selection_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=inspect_record(fixture(tmp))
            generator=Mock(side_effect=ValueError('synthetic rendering failure'))
            dialog=RecordComparisonDialog(root=tmp,scanner=lambda *a:[record],generator=generator)
            dialog.show();self.wait(lambda:not dialog.busy);dialog.table.item(0,0).setCheckState(Qt.Checked)
            dialog.generate();self.wait(lambda:not dialog.busy)
            self.assertIn('synthetic rendering failure',dialog.status.text());self.assertTrue(dialog.generate_button.isEnabled())
            self.assertEqual(len(dialog.checked),1);dialog.close()

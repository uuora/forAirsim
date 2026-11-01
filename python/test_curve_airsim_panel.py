"""Selection/history regressions; no simulator or native window is started."""
import json
import queue
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from curve_airsim_experiment_panel import ExperimentPanel, TITLES, catalogue, curve_order_note, exact_selections, history_records, launch_command


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def make_run(root,name,selected=None,reports=None):
    directory=Path(root)/name/"replay"
    plans=[dict(group_id=key,seed=1101,parameters=[0,1.15,0,5.5]) for key in TITLES]
    manifest=dict(probe_only=False,plans_document=dict(schema_version=1,plans=plans),started_at_utc="2026-10-29T00:00:00Z")
    if selected is not None:
        manifest["selected_group_ids"]=selected
    write(directory/"manifest.json",manifest)
    write(directory/"rows.json",reports or [])
    return directory


class PanelSelectionTests(unittest.TestCase):
    def test_curve_order_comes_from_parameters_not_ordinary_method_name(self):
        zero=dict(group_id='hermite_ordinary',family='single_bump',parameters=[0,0,0,5.5])
        shaped=dict(zero,parameters=[0,1.15,0,5.5])
        self.assertIn('五次',curve_order_note(zero))
        self.assertIn('至多六次',curve_order_note(shaped))
        self.assertNotIn('五次',curve_order_note(shaped))
        self.assertEqual(curve_order_note({'group_id':'hermite_ordinary'}),'次数未核验')

    def test_queued_continuation_uses_frozen_seed_and_method(self):
        view=SimpleNamespace(busy=False,continuous_remaining=1,
            campaign=dict(status='RUNNING',method='hermite_abom',seeds=[1101,1102],runs=[{}]),
            method=Mock(),seed_value=Mock(),generate_targets=Mock())
        ExperimentPanel._continue_campaign(view)
        view.seed_value.set.assert_called_once_with('1102')
        view.method.set.assert_called_once_with(TITLES['hermite_abom'])
        view.generate_targets.assert_called_once_with(True,run_after=True)

    def test_stop_between_episodes_cancels_queued_generation(self):
        view=SimpleNamespace(busy=False,continuous_remaining=1,
            campaign=dict(status='RUNNING',method='hermite_abom',seeds=[1101,1102],runs=[{}]),
            method=Mock(),seed_value=Mock(),generate_targets=Mock(),_save_campaign=Mock(),_log=Mock())
        ExperimentPanel.stop_continuous(view)
        ExperimentPanel._continue_campaign(view)
        self.assertEqual(view.campaign['status'],'STOPPED')
        view.generate_targets.assert_not_called()

    def test_idle_stop_does_not_rewrite_completed_campaign(self):
        view=SimpleNamespace(campaign=dict(status='COMPLETED'),_save_campaign=Mock(),_log=Mock())
        ExperimentPanel.stop_continuous(view)
        self.assertEqual(view.campaign['status'],'COMPLETED')
        view._save_campaign.assert_not_called()

    def test_launch_command_contains_only_the_one_explicit_group(self):
        with TemporaryDirectory() as folder:
            plans=Path(folder)/"plans.json"; config=Path(folder)/"config.json"
            write(plans,dict(schema_version=1,plans=[dict(group_id=k) for k in TITLES]))
            write(config,{})
            command=launch_command("hermite_abom",plans,config,Path(folder)/"output",shell="powershell.exe")
            index=command.index("-Groups")
            self.assertEqual(command[index+1],"hermite_abom")
            self.assertEqual([value for value in command if value in TITLES],["hermite_abom"])
            self.assertIn("-ShowWindow",command)
            self.assertNotIn("-AllGroups",command)

    def test_unknown_group_is_rejected_without_reading_or_launching(self):
        with self.assertRaises(ValueError):
            launch_command("not_a_method","missing.json","missing.json","output")

    def test_invalid_catalogue_is_rejected(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"plans.json"
            write(path,dict(schema_version=1,plans=[dict(group_id="hermite_abom")]*6))
            with self.assertRaises(ValueError): catalogue(path)

    def test_history_does_not_invent_unselected_execution_slots(self):
        with TemporaryDirectory() as folder:
            path=make_run(folder,"one",["hermite_abom"],
                [dict(group_id="hermite_abom",status="COMPLETED",metrics=dict(position_rmse_m=.3,control_error_gates_passed=False))])
            records,errors=history_records([path,path.parent])
            self.assertFalse(errors)
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]["group_id"],"hermite_abom")
            self.assertFalse(records[0]["control_passed"])

    def test_exact_pair_selection_does_not_promote_other_groups_from_a_batch(self):
        with TemporaryDirectory() as folder:
            old=make_run(folder,"old")
            new=make_run(folder,"new",["hermite_abom_improved_candidate"])
            choices=exact_selections([
                dict(run_path=str(old),group_id="hermite_ordinary"),
                dict(run_path=str(new),group_id="hermite_abom_improved_candidate"),
            ])
            self.assertEqual(choices,[dict(run_path=str(old.resolve()),group_ids=["hermite_ordinary"]),
                dict(run_path=str(new.resolve()),group_ids=["hermite_abom_improved_candidate"])])

    def test_repeat_execution_is_retained_and_legacy_unrun_is_explicit(self):
        with TemporaryDirectory() as folder:
            old=make_run(folder,"old",reports=[dict(group_id="hermite_abom",status="FAILED")])
            new=make_run(folder,"new",["hermite_abom"],[dict(group_id="hermite_abom",status="COMPLETED",metrics={})])
            rows,errors=history_records([old,new])
            self.assertFalse(errors)
            self.assertEqual(len(rows),7)
            self.assertEqual(sum(row["group_id"]=="hermite_abom" for row in rows),2)
            self.assertEqual(sum(row["status"]=="NOT_RUN" for row in rows),5)

    def test_running_batch_distinguishes_active_and_pending_slots(self):
        with TemporaryDirectory() as folder:
            path=make_run(folder,'active',['hermite_ordinary','hermite_abom'])
            manifest=json.loads((path/'manifest.json').read_text());manifest['status']='RUNNING';write(path/'manifest.json',manifest)
            (path/'hermite_ordinary').mkdir()
            rows,errors=history_records([path])
            self.assertFalse(errors)
            self.assertEqual({r['group_id']:r['status'] for r in rows},{'hermite_ordinary':'RUNNING','hermite_abom':'PENDING'})

    def test_launcher_saves_late_output_without_a_ui_pipe_reader(self):
        with TemporaryDirectory() as folder:
            view=SimpleNamespace(busy=False,campaign=None,workspace=Path(folder),messages=queue.Queue(),
                selected_group=lambda:"hermite_abom",plans_path=SimpleNamespace(get=lambda:"plans.json"),
                config_path=SimpleNamespace(get=lambda:"config.json"),show_window=SimpleNamespace(get=lambda:False),
                status=SimpleNamespace(set=lambda value:None),_log=lambda value:None)
            view._set_busy=lambda value:setattr(view,"busy",value)
            actual_popen=subprocess.Popen
            children=[]
            def capture_popen(*args,**kwargs):
                process=actual_popen(*args,**kwargs)
                children.append(process)
                return process
            command=[sys.executable,"-u","-c","import time; print('started'); time.sleep(.15); print('finished')"]
            with patch("curve_airsim_experiment_panel.launch_command",return_value=command), \
                    patch("curve_airsim_experiment_panel.subprocess.Popen",side_effect=capture_popen):
                ExperimentPanel.start_run(view)
                # No Tk loop drains the queue while the child runs.
                deadline=time.monotonic()+5
                result=None
                while time.monotonic()<deadline:
                    try:
                        kind,value=view.messages.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if kind in ("run_done","error"):
                        result=(kind,value)
                        break
                self.assertIsNotNone(result)
                self.assertEqual(result[0],"run_done",result)
                self.assertEqual(result[1]["code"],0)
                self.assertIsNone(children[0].stdout)  # stdout belongs to a file, not the panel.
                log_files=list((Path(folder)/"reports").glob("*.launcher.log"))
                self.assertEqual(len(log_files),1)
                self.assertEqual(log_files[0].read_text(encoding="utf-8").splitlines(),["started","finished"])


if __name__=="__main__":
    unittest.main()

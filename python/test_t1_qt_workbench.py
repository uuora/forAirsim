"""Check replay provenance, missing slots and asynchronous Qt state."""
import json,os,tempfile,time,unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage,QColor
from t1_qt_evidence import ROOT,sha256,load_campaign,load_frames,validate_frame_image,latest_analysis
from t1_qt_dashboard import ExperimentDashboard,FrameImage
from vlx_gui_qt import VLXSeekGUI


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="qt_evidence_test_",dir=ROOT/"logs")
        self.folder=Path(self.tmp.name).resolve()
        assert self.folder.is_relative_to((ROOT/"logs").resolve())
        self.run=self.folder/"run";(self.run/"flight").mkdir(parents=True)
        self.image=self.run/"flight/frame.png"
        q=QImage(24,24,QImage.Format_RGB32);q.fill(QColor("#eeeeee"));assert q.save(str(self.image))
        frames=[]
        for i in (0,2):
            frames.append(dict(slot_index=i,phase_id="visible",frame_id=f"main-{i}",
                capture_received_host_s=100+i*.4,media=dict(rgb=dict(path=str(self.image),sha256=sha256(self.image))),
                observer_result=dict(semantic_frame=dict(candidates=[dict(bbox_xyxy_pixels=[2,2,18,18])]),
                    temporal_identity=dict(selected_candidate=dict(bbox_xyxy_pixels=[2,2,18,18]) if i==0 else None),
                    semantic_poll=dict(reason="SEMANTIC_RESPONSE_NOT_READY")),
                motion_gate=dict(reason="NO_CURRENT_DESIGNATED_TARGET")))
        self.report=self.run/"flight/report.json"
        self.write(self.report,dict(session_id="test_run",frames=frames,
            slots=[dict(slot_index=i,phase_id="visible",offset_s=i*.4,status="CAPTURED" if i!=1 else "MISSED") for i in range(3)]))
        base=dict(measured_epoch_monotonic_s=100.,measured_frame_count=2,acquisition_frame_count=1,
            measured_identity_counts=dict(CORRECT=1,NO_OUTPUT=1),
            independent_integration_qualification="PASS_CONTROLLED_DEVELOPMENT_SCOPE",
            frame_rows=[dict(frame_id="initial",readback_s=99.,actual_phase="initial_acquisition",identity=dict(status="NO_OUTPUT"))]+[
                dict(frame_id=f"main-{i}",readback_s=100+i*.4,actual_phase="visible",identity=dict(status="CORRECT" if i==0 else "NO_OUTPUT")) for i in (0,2)])
        source=dict(session_id="test_run",run=str(self.run),base_evidence=base,mode="SHORT_XY",
            endpoint=dict(net_xy_displacement_m=.03,start_state=dict(position_world_ned_m=[0.,0.,-3.])),
            endpoint_scale_status="PASS_ENDPOINT_MOTION_SCALE",plan_check=dict(expected_main_slots=3),
            sampled_state_rows=[dict(host_receipt_monotonic_s=100.,position_world_ned_m=[0.,0.,-3.])],
            frame_timing_rows=[],timing_summary=dict(historical_full=dict(counts=dict(TIMELY=2))),
            input_receipts=[dict(path=str(self.report),sha256=sha256(self.report))])
        self.analysis=self.folder/"analysis.json"
        self.write(self.analysis,dict(kind="T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS",
            formal_efficacy_results=False,runs=[source]))
        self.widgets=[]

    def tearDown(self):
        for widget in self.widgets:
            self.wait_for(lambda:not getattr(widget,"busy",False))
            widget.close()
        self.app.processEvents()
        assert self.folder.is_relative_to((ROOT/"logs").resolve())
        self.tmp.cleanup()

    def write(self,path,value):
        path.write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")

    def wait_for(self,condition,timeout=5):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.app.processEvents()
            if condition():return
            time.sleep(.01)
        self.fail("Qt asynchronous state did not finish")

    def campaign(self):
        return load_campaign(self.analysis)

    def test_initial_frames_are_not_counted_as_measured_identity_points(self):
        run=self.campaign().runs[0]
        self.assertEqual(len(run.identity_points),2)
        self.assertEqual(run.counts,dict(CORRECT=1,NO_OUTPUT=1))

    def test_missing_planned_slot_is_retained_without_an_old_image(self):
        frames=load_frames(self.campaign().runs[0])
        self.assertEqual(len(frames),3)
        self.assertEqual(frames[1].identity,"MISSING")
        self.assertIsNone(frames[1].image)

    def test_report_digest_mismatch_prevents_pairing(self):
        run=self.campaign().runs[0]
        self.report.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):load_frames(run)

    def test_changed_image_clears_the_previous_overlay(self):
        frame=load_frames(self.campaign().runs[0])[0]
        canvas=FrameImage();self.widgets.append(canvas)
        self.assertTrue(canvas.set_frame(frame))
        self.image.write_bytes(b"changed")
        self.assertFalse(canvas.set_frame(frame))
        self.assertIsNone(canvas.annotated)

    def test_formal_or_unrelated_json_does_not_become_development_evidence(self):
        data=json.loads(self.analysis.read_text(encoding="utf-8"))
        data["formal_efficacy_results"]=True;self.write(self.analysis,data)
        with self.assertRaises(ValueError):load_campaign(self.analysis)

    def test_outside_project_path_is_refused_before_reading(self):
        with self.assertRaises(ValueError):load_campaign(ROOT.parent/"outside_project_analysis.json")

    def test_registered_latest_analysis_digest_must_match(self):
        config=self.folder/"configs";config.mkdir()
        self.write(config/"t1_completion_protocol_v1.json",dict(latest_scaled_motion_development=dict(
            analysis_path=str(self.analysis),analysis_sha256=sha256(self.analysis))))
        self.assertEqual(latest_analysis(self.folder),self.analysis)
        self.analysis.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):latest_analysis(self.folder)

    def test_unmeasured_failed_attempt_has_planned_slots_and_no_fake_clock(self):
        data=json.loads(self.analysis.read_text(encoding="utf-8"));source=data["runs"][0]
        report=json.loads(self.report.read_text(encoding="utf-8"));report["frames"]=[];self.write(self.report,report)
        source["input_receipts"][0]["sha256"]=sha256(self.report)
        source["base_evidence"].update(measured_epoch_monotonic_s=None,measured_frame_count=0,
            measured_identity_counts={},frame_rows=[])
        source["endpoint"].update(start_state=None,net_xy_displacement_m=None)
        self.write(self.analysis,data)
        campaign=self.campaign();frames=load_frames(campaign.runs[0])
        self.assertIsNone(campaign.runs[0].epoch)
        self.assertEqual(len(frames),3)
        self.assertTrue(all(f.identity=="MISSING" and not f.actual_time for f in frames))
        dashboard=ExperimentDashboard();self.widgets.append(dashboard)
        dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        self.assertIn("计划时刻",dashboard.frame_label.text())

    def partial_ledger(self):
        data=json.loads(self.analysis.read_text(encoding="utf-8"));source=data["runs"][0]
        report=json.loads(self.report.read_text(encoding="utf-8"));ledger=self.run/"flight/partial.jsonl"
        ledger.write_text("".join(json.dumps(f)+"\n" for f in report["frames"]),encoding="utf-8")
        report["frames"]=[];self.write(self.report,report)
        source["input_receipts"][0]["sha256"]=sha256(self.report)
        source["incomplete_frame_ledger"]=dict(path=str(ledger),sha256=sha256(ledger))
        source["base_evidence"]["measured_identity_counts"]={"UNSCORED":2}
        for row in source["base_evidence"]["frame_rows"]:
            row["identity"]["status"]="UNSCORED"
        self.write(self.analysis,data)
        return ledger

    def test_partial_ledger_is_replayed_as_unscored_with_missing_slots_retained(self):
        self.partial_ledger();run=self.campaign().runs[0];frames=load_frames(run)
        self.assertEqual([f.identity for f in frames],["UNSCORED","MISSING","UNSCORED"])
        self.assertTrue(frames[0].actual_time)
        self.assertEqual(run.counts,{"UNSCORED":2})

    def test_changed_partial_ledger_prevents_replay(self):
        ledger=self.partial_ledger();run=self.campaign().runs[0]
        ledger.write_text("{}\n",encoding="utf-8")
        with self.assertRaises(ValueError):load_frames(run)

    def test_async_invalid_load_keeps_the_previous_campaign(self):
        dashboard=ExperimentDashboard();self.widgets.append(dashboard)
        dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        old=dashboard.campaign
        invalid=self.folder/"invalid.json";invalid.write_text("{}",encoding="utf-8")
        dashboard.load_analysis(invalid);self.wait_for(lambda:not dashboard.busy)
        self.assertIs(dashboard.campaign,old)
        self.assertIsNotNone(dashboard.last_error)

    def test_slider_and_playback_stop_at_last_saved_slot(self):
        dashboard=ExperimentDashboard();self.widgets.append(dashboard)
        dashboard.load_analysis(self.analysis)
        self.wait_for(lambda:not dashboard.busy and bool(dashboard.frames))
        dashboard.slider.setValue(1)
        self.assertIsNone(dashboard.image.annotated)
        dashboard.slider.setValue(2);dashboard.toggle_play()
        self.assertTrue(dashboard.timer.isActive())
        dashboard.slider.setValue(2);dashboard.next_frame()
        self.assertFalse(dashboard.timer.isActive())

    def test_single_image_page_defaults_to_frozen_balloon_profile(self):
        window=VLXSeekGUI(auto_load=False,initialize_api=False);self.widgets.append(window)
        self.assertEqual(window.detector_profile.currentData(),"balloon")
        self.assertEqual(window.description_input.text(),"带白色横条的红色气球")
        self.assertTrue(window.load_image_file(str(self.image)))
        window.set_busy(True,"测试后台任务")
        self.assertFalse(window.detector_profile.isEnabled())
        self.assertFalse(window.clear_results_btn.isEnabled())
        self.assertTrue(all(not b.isEnabled() for b in window.example_buttons))
        window.set_description("最右侧的红色气球")
        self.assertEqual(window.description_input.text(),"带白色横条的红色气球")
        window.set_busy(False,"完成")

    def test_long_saved_image_name_does_not_clip_single_image_controls(self):
        window=VLXSeekGUI(auto_load=False,initialize_api=False);self.widgets.append(window)
        window.tabs.setCurrentIndex(1);window.resize(1050,700);window.show()
        window.image_info.setText("t1_source_occlusion_v8_20261006_f006_rgb.png · 640×480 · 拖动鼠标添加候选框")
        window.vlx_info.setText("已返回定位框：带白色横条的红色气球 · 0.91秒")
        for _ in range(4):self.app.processEvents()
        self.assertEqual(window.image_controls_scroll.horizontalScrollBar().maximum(),0)
        self.assertTrue(window.request_note.isVisible())


if __name__=="__main__":unittest.main()

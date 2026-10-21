"""Render Qt, replay saved evidence and optionally make one real VLX GUI call."""
import argparse,csv,json,time
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
from PyQt5.QtGui import QFontMetrics,QFontDatabase
from vlx_gui_qt import VLXSeekGUI
from t1_qt_evidence import ROOT,sha256


def verify(output,live_vlx=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"logs"):raise ValueError("PROJECT_LOG_OUTPUT_REQUIRED")
    output.mkdir(exist_ok=True)
    receipt_path=output/"qt_verification.json"
    if receipt_path.exists():raise FileExistsError("NEW_QT_VERIFICATION_REQUIRED")
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion")
    warnings=[];result=dict(kind="T1_QT_WORKBENCH_VERIFICATION",
        started_utc=datetime.now(timezone.utc).isoformat(),platform=app.platformName(),
        native_desktop_interaction_tested=False,live_vlx_requested=live_vlx,
        simulator_calls=0,flight_requests=0,api_request_mode="image_text",
        proposals_transmitted=False,formal_efficacy_results=False)
    window=VLXSeekGUI(auto_load=False,initialize_api=live_vlx);window.show()
    def wait(condition,timeout=45):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            app.processEvents()
            if condition():return
            time.sleep(.01)
        raise TimeoutError("QT_JOB_DID_NOT_COMPLETE_WITHIN_VERIFICATION_BOUND")
    def capture(name):
        app.processEvents();p=output/name
        if p.exists():raise FileExistsError("NEW_QT_RENDER_REQUIRED")
        if not window.grab().save(str(p)):raise ValueError("QT_RENDER_FAILED")
        return dict(path=str(p),sha256=sha256(p),size_wh=[window.width(),window.height()])
    renders=[]
    try:
        with patch.object(QMessageBox,"warning",lambda *args:warnings.append("GUI_WARNING_DISPLAYED")):
            window.research.load_latest()
            wait(lambda:not window.research.busy and bool(window.research.frames))
            campaign=window.research.campaign
            result["analysis"]=dict(path=str(campaign.path),sha256=campaign.digest)
            index=next((i for i,r in enumerate(campaign.runs) if r.source.get("source_perturbation",{}).get("condition")=="occlusion"),0)
            window.research.session_combo.setCurrentIndex(index)
            wait(lambda:not window.research.busy and bool(window.research.frames))
            dashboard=window.research
            result["replay"]=dict(session_id=dashboard.run.session_id,planned_slots=len(dashboard.frames),
                measured=dashboard.run.measured,counts=dashboard.run.counts,net_mm=dashboard.run.net_mm,
                original_full_late=dashboard.run.source["timing_summary"]["historical_full"]["counts"].get("LATE",0))
            hidden=next((i for i,f in enumerate(dashboard.frames) if f.phase=="hidden"),0)
            dashboard.slider.setValue(hidden);dashboard.plots.setCurrentIndex(1)
            renders.append(capture("qt_verified_identity.png"))
            visible=next((i for i,f in enumerate(dashboard.frames) if f.identity=="CORRECT"),0)
            dashboard.slider.setValue(visible);dashboard.plots.setCurrentIndex(0)
            renders.append(capture("qt_verified_trajectory.png"))
            dashboard.plots.setCurrentIndex(2);renders.append(capture("qt_verified_timing.png"))
            window.resize(1050,700);dashboard.plots.setCurrentIndex(1)
            renders.append(capture("qt_verified_compact.png"));window.resize(1440,880)
            dashboard.slider.setValue(visible);app.processEvents()
            exported_image=output/"qt_exported_frame.png";exported_csv=output/"qt_exported_frames.csv"
            with patch.object(QFileDialog,"getSaveFileName",return_value=(str(exported_image),"PNG")):
                dashboard.export_image_btn.click()
            with patch.object(QFileDialog,"getSaveFileName",return_value=(str(exported_csv),"CSV")):
                dashboard.export_csv_btn.click()
            with exported_csv.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.reader(stream))
            assert len(rows)==len(dashboard.frames)+1
            assert exported_image.is_file()
            result["exports"]=[dict(path=str(p),sha256=sha256(p)) for p in (exported_image,exported_csv)]
            before=dashboard.slider.value();dashboard.play_btn.click()
            wait(lambda:dashboard.slider.value()>before,3)
            dashboard.play_btn.click();assert not dashboard.timer.isActive()
            result["playback_advances_and_pauses"]=True
            window.tabs.setCurrentIndex(1)
            image_path=dashboard.frames[visible].image
            assert window.load_image_file(str(image_path))
            yolo_results=[];vlx_results=[]
            window.run_yolo();window.detection_thread.finished.connect(yolo_results.append)
            wait(lambda:not window.busy and not window.detection_thread.isRunning())
            assert window.yolo_boxes and yolo_results and not warnings
            result["single_image"]=dict(path=str(image_path),sha256=sha256(image_path),
                detector_profile=window.detector_profile.currentData(),detector_device="cpu",
                detector_candidates=window.yolo_boxes)
            if live_vlx:
                assert window.vlx is not None
                window.run_vlx();window.vlx_thread.finished.connect(vlx_results.append)
                wait(lambda:not window.busy and not window.vlx_thread.isRunning())
                assert vlx_results and not warnings
                selected=vlx_results[0].get("result_bbox_list",[])
                assert selected and window.canvas.vlx_box is not None
                result["single_image"]["vlx"]=dict(model=window.vlx.model,
                    result_bbox_list=selected,answer=vlx_results[0].get("answer",""),
                    gui_call_wall_s=vlx_results[0]["gui_call_wall_s"],
                    completed_response_count=window.vlx.total_requests,
                    request_mode=window.vlx.request_mode,proposals_transmitted=window.vlx.proposals_transmitted)
            renders.append(capture("qt_verified_single_image.png"))
            metrics=QFontMetrics(window.font())
            result["font"]=dict(family=window.font().family(),families=QFontDatabase().families(),
                chinese_glyphs_present=all(metrics.inFontUcs4(ord(c)) for c in "任务一轨迹遮挡正确输出气球"))
            assert result["font"]["chinese_glyphs_present"]
            result.update(status="PASS_OFFSCREEN_QT_WITH_REAL_SINGLE_IMAGE_MODELS",renders=renders,
                warnings=warnings,visual_review=dict(status="PENDING"))
    except Exception as exc:
        result.update(status="FAIL_QT_VERIFICATION",error_type=type(exc).__name__,
            renders=renders,warnings=warnings,visual_review=dict(status="PENDING"))
        raise
    finally:
        result["finished_utc"]=datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if not window.research.has_running_job():
            window.close();app.processEvents()
    print(result["status"],"renders",len(renders),"flight_requests",0)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--live-vlx",action="store_true");a=p.parse_args();verify(a.output,a.live_vlx)

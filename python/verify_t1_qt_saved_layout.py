"""Render the current Qt layout from saved, hash-checked real model results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from PyQt5.QtWidgets import QApplication
from vlx_gui_qt import VLXSeekGUI
from t1_qt_evidence import ROOT, sha256


def verify(prior, output, analysis=None):
    prior, output = Path(prior).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT/"logs") or output.exists():
        raise ValueError("NEW_PROJECT_LOG_LAYOUT_OUTPUT_REQUIRED")
    source = json.loads(prior.read_text(encoding="utf-8"))
    if source["status"] != "PASS_OFFSCREEN_QT_WITH_REAL_SINGLE_IMAGE_MODELS":
        raise ValueError("REAL_MODEL_GUI_VERIFICATION_REQUIRED")
    single = source["single_image"]
    if sha256(single["path"]) != single["sha256"]:
        raise ValueError("SAVED_SINGLE_IMAGE_CHANGED")
    for record in source["renders"]+source["exports"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("PRIOR_GUI_ARTIFACT_CHANGED")
    output.mkdir()
    app = QApplication.instance() or QApplication([]); app.setStyle("Fusion")
    window = VLXSeekGUI(auto_load=False, initialize_api=False); window.show()
    result = dict(kind="T1_QT_SAVED_EVIDENCE_LAYOUT_VERIFICATION", started_utc=datetime.now(timezone.utc).isoformat(),
        platform=app.platformName(), native_desktop_interaction_tested=False,
        prior_real_model_receipt=dict(path=str(prior), sha256=sha256(prior)),
        live_vlx_requests=0, detector_calls=0, flight_requests=0, simulator_calls=0,
        saved_model_result_replayed=True, formal_efficacy_results=False, renders=[])
    def wait(condition):
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            app.processEvents()
            if condition(): return
            time.sleep(.01)
        raise TimeoutError("QT_SAVED_LAYOUT_LOAD_TIMEOUT")
    def capture(name):
        for _ in range(5): app.processEvents()
        path = output/name
        if not window.grab().save(str(path)): raise ValueError("QT_LAYOUT_RENDER_FAILED")
        result["renders"].append(dict(path=str(path), sha256=sha256(path), size_wh=[window.width(), window.height()]))
    try:
        if analysis is None: window.research.load_latest()
        else: window.research.load_analysis(Path(analysis))
        wait(lambda: not window.research.busy and bool(window.research.frames))
        dashboard = window.research
        result["analysis"] = dict(path=str(dashboard.campaign.path), sha256=dashboard.campaign.digest)
        if dashboard.campaign.runs:
            dashboard.session_combo.setCurrentIndex(len(dashboard.campaign.runs)-1)
            wait(lambda: not dashboard.busy and bool(dashboard.frames))
        result["replay"] = dict(session_id=dashboard.run.session_id, main_slots=len(dashboard.frames),
            measured=dashboard.run.measured, counts=dashboard.run.counts, net_mm=dashboard.run.net_mm)
        dashboard.slider.setValue(len(dashboard.frames)-1)
        for index, name in [(0, "qt_current_trajectory.png"), (1, "qt_current_identity.png"), (2, "qt_current_timing.png")]:
            dashboard.plots.setCurrentIndex(index); capture(name)
        window.resize(1050, 700); dashboard.plots.setCurrentIndex(1); capture("qt_current_compact.png")
        window.resize(1440,880)
        for index,run in enumerate(dashboard.campaign.runs):
            if not run.source.get("incomplete_record_notice"):continue
            dashboard.session_combo.setCurrentIndex(index)
            wait(lambda:not dashboard.busy and bool(dashboard.frames))
            dashboard.slider.setValue(0);dashboard.plots.setCurrentIndex(1)
            capture(f"qt_current_incomplete_{index+1}.png")
        window.resize(1050,700)
        window.tabs.setCurrentIndex(1)
        assert window.load_image_file(single["path"])
        window.on_yolo_finished(single["detector_candidates"])
        vlx = single["vlx"]
        window.on_vlx_finished(dict(result_bbox_list=vlx["result_bbox_list"], answer=vlx["answer"], gui_call_wall_s=vlx["gui_call_wall_s"]))
        window.statusBar().showMessage("已保存单图结果回放 · 本次未调用检测或云端服务")
        capture("qt_current_single_compact.png")
        assert window.image_controls_scroll.horizontalScrollBar().maximum() == 0
        window.resize(1440, 880); capture("qt_current_single_image.png")
        assert window.image_controls_scroll.horizontalScrollBar().maximum() == 0
        result.update(status="PASS_OFFSCREEN_SAVED_EVIDENCE_LAYOUT", horizontal_scroll_maximum=0,
            visual_review=dict(status="PENDING"), source_files=[dict(path=str(ROOT/"python"/name), sha256=sha256(ROOT/"python"/name))
                for name in ("vlx_gui_qt.py", "t1_qt_dashboard.py", "t1_qt_evidence.py", "verify_t1_qt_saved_layout.py")])
    except Exception as exc:
        result.update(status="FAIL_SAVED_EVIDENCE_LAYOUT", error_type=type(exc).__name__, visual_review=dict(status="PENDING"))
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output/"layout_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if not window.research.has_running_job(): window.close(); app.processEvents()
    print(result["status"], "renders", len(result["renders"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args(); verify(args.prior, args.output, args.analysis)

"""Three-request, saved-RGB YOLO/VLX latency diagnostic; no simulation or flight.

The observer factory loads authorized API configuration internally. Only closed
numeric/category/hash fields are saved; endpoint, credentials, raw response text
and exception messages never enter the output. Replay host receipts are newly
created diagnostic clocks, not fresh camera observations or flight evidence.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from t1_continuous_semantic_observer import create_observer
from t1_semantic_perception import PolicyConfig, SemanticRequest, box_iou

DESCRIPTION = "带白色横条的红色气球"
CONFIG_FIELDS = frozenset({"status", "class_names", "backend", "weights", "weights_sha256",
    "training_receipt", "training_receipt_sha256", "imgsz", "confidence", "nms_iou", "device",
    "vlx_adapter_injection", "designated_identity_established_by_detection",
    "flight_integration_verified", "runtime_diagnostic", "interpretation"})
MODEL_FIELDS = frozenset({"weights_config_path", "weights_config_sha256", "weights_path", "weights_sha256",
    "training_receipt_sha256", "confidence", "nms_iou", "imgsz", "device", "model",
    "request_mode", "proposals_transmitted"})
USAGE_FIELDS = frozenset({"total_requests", "total_prompt_tokens", "total_completion_tokens",
    "total_tokens", "total_took_ms", "average_took_ms"})


class DiscardOutput:
    """Do not retain library output, which may include a request exception URL."""
    def write(self, text):
        return len(text)
    def flush(self):
        pass


def digest(path):
    path = Path(path)
    if path.name.casefold().startswith(".env"):
        raise ValueError("Credential files are outside diagnostic inputs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def safe_error_category(error):
    """Type-only classification; never inspect or serialize exception strings."""
    import requests
    visited = set()
    current = error
    for _ in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        for cls, category in ((requests.exceptions.ReadTimeout, "READ_TIMEOUT"),
                (requests.exceptions.ConnectTimeout, "CONNECT_TIMEOUT"),
                (requests.exceptions.Timeout, "REQUEST_TIMEOUT"),
                (requests.exceptions.HTTPError, "HTTP_STATUS_ERROR"),
                (requests.exceptions.ConnectionError, "CONNECTION_ERROR"),
                (requests.exceptions.RequestException, "REQUEST_TRANSPORT_ERROR")):
            if isinstance(current, cls):
                return category
        current = current.__cause__ or current.__context__
    if isinstance(error, (KeyError, TypeError, ValueError)):
        return "VALUE_OR_RESPONSE_SCHEMA_ERROR"
    if isinstance(error, RuntimeError):
        return "BACKEND_OR_SERVICE_RESPONSE_FAILURE"
    if isinstance(error, OSError):
        return "LOCAL_INPUT_OR_RUNTIME_ERROR"
    return "OTHER_ERROR"


def usage(client):
    raw = client.get_usage_summary()
    return {key: float(value) if isinstance(value, float) else int(value)
        for key, value in raw.items() if key in USAGE_FIELDS and number(value)}


def association_summary(result, frame):
    """Source-box diagnostic only; does not grant current-frame identity output."""
    boxes = result.get("result_bbox_list") if isinstance(result, dict) else None
    if not isinstance(boxes, list):
        return {"bbox_count": None, "association_status": "INVALID_BBOX_LIST"}
    if len(boxes) != 1:
        return {"bbox_count": len(boxes), "association_status": "NO_SINGLE_SEMANTIC_BOX"}
    try:
        box = tuple(boxes[0][key] for key in ("xmin", "ymin", "xmax", "ymax"))
        if not all(number(value) for value in box) or not (
                0 <= box[0] < box[2] <= frame.width and 0 <= box[1] < box[3] <= frame.height):
            raise ValueError("Invalid box")
        scores = sorted([box_iou(box, candidate.bbox_xyxy_pixels) for candidate in frame.candidates], reverse=True)
    except (KeyError, TypeError, ValueError):
        return {"bbox_count": 1, "association_status": "INVALID_SEMANTIC_BOX"}
    config = PolicyConfig()
    top, second = (scores[0] if scores else None), (scores[1] if len(scores) > 1 else None)
    unique = bool(scores and top >= config.source_association_min_iou and
        (second is None or (second < config.source_association_min_iou and top-second >= config.association_margin)))
    return {"bbox_count": 1, "association_status": "UNIQUE_SOURCE_YOLO_ASSOCIATION" if unique else "SOURCE_ASSOCIATION_UNRESOLVED",
        "source_top_iou": top, "source_second_iou": second,
        "source_minimum_iou": config.source_association_min_iou,
        "association_margin": config.association_margin}


def one_request(backend, rgb, image_sha, session, index):
    receipt = time.monotonic()
    row = {"request_index": index, "status": "PREPARING", "replay_receipt_monotonic_s": receipt,
        "request_mode": "image_text", "proposals_transmitted": False,
        "reply_binding_provenance": "CLIENT_SIDE_BINDING; NOT_SERVER_ECHO"}
    try:
        frame = backend.detect_frame(rgb, session_id=session, target_generation=1,
            frame_id=f"{session}_f{index:03d}", frame_sequence=index,
            readback_receipt_monotonic_s=receipt, expected_image_sha256=image_sha)
    except Exception as error:
        return dict(row, status="LOCAL_PREPARATION_FAILED", error_category=safe_error_category(error),
            preparation_wall_s=time.monotonic()-receipt, backend_reply_returned=False)
    submitted = time.monotonic()
    request = SemanticRequest(f"diagnostic-{index:03d}", session, 1, DESCRIPTION,
        hashlib.sha256(DESCRIPTION.encode()).hexdigest(), submitted, frame)
    binding_json = json.dumps(request.binding(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    before = usage(backend.vlx_client)
    row.update(status="REQUESTED", frame_id=frame.frame_id, frame_sequence=frame.frame_sequence,
        yolo_candidate_count=len(frame.candidates), candidates_ready_monotonic_s=frame.candidates_ready_monotonic_s,
        candidate_preparation_wall_s=frame.candidates_ready_monotonic_s-receipt,
        source_age_at_submit_s=submitted-receipt, submitted_monotonic_s=submitted,
        request_image_sha256=frame.image_sha256,
        request_binding_sha256=hashlib.sha256(binding_json.encode()).hexdigest())
    reply = None
    error_category = None
    try:
        reply = backend(request)
    except Exception as error:
        error_category = safe_error_category(error)
    returned = time.monotonic()
    after = usage(backend.vlx_client)
    wall = returned-submitted
    policy = PolicyConfig()
    row.update(status="BACKEND_REPLY_RETURNED" if reply is not None else "BACKEND_REQUEST_FAILED",
        backend_reply_returned=reply is not None, error_category=error_category,
        returned_monotonic_s=returned, request_wall_s=wall,
        response_timeout_2s_met=wall <= policy.response_timeout_s,
        source_age_at_return_s=returned-receipt,
        semantic_source_age_2s_met=returned-receipt <= policy.semantic_max_age_s,
        client_parsed_response_counter_delta=after.get("total_requests", 0)-before.get("total_requests", 0),
        usage_counter_deltas={key: after.get(key, 0)-before.get(key, 0)
            for key in USAGE_FIELDS-{ "average_took_ms" } if key in after and key in before})
    if reply is not None:
        binding = getattr(reply, "binding", None)
        completed = getattr(reply, "completed_monotonic_s", None)
        row.update(reply_binding_status="MATCH" if binding == request.binding() else "MISMATCH",
            reply_image_sha256=binding.get("image_sha256") if isinstance(binding, dict) and
                isinstance(binding.get("image_sha256"), str) and len(binding["image_sha256"]) == 64 else None,
            backend_completion_monotonic_s=completed if number(completed) else None,
            backend_completion_wall_s=completed-submitted if number(completed) else None,
            completion_clock_within_request_return=number(completed) and submitted <= completed <= returned)
        row.update(association_summary(getattr(reply, "result", None), frame))
    else:
        row.update(reply_binding_status="NO_REPLY", reply_image_sha256=None,
            bbox_count=None, association_status="NO_REPLY")
    return row


def profile(rgb, detector_config, output, requests=3):
    if isinstance(requests, bool) or not isinstance(requests, int) or not 1 <= requests <= 3:
        raise ValueError("Diagnostic request count must be 1 to 3")
    rgb, detector_config, output = Path(rgb).resolve(), Path(detector_config).resolve(), Path(output).resolve()
    image_sha, config_sha = digest(rgb), digest(detector_config)
    config = json.loads(detector_config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not set(config).issubset(CONFIG_FIELDS):
        raise ValueError("Detector config exceeds the closed diagnostic field allowlist")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshots"
    snapshot.mkdir()
    sources = [Path(__file__).resolve()]+[Path(__file__).resolve().with_name(name) for name in (
        "t1_continuous_semantic_observer.py", "t1_semantic_perception.py", "vlx_api_client.py",
        "specialist_balloon_detector.py", "train_t1_balloon_yolo.py")]
    source_rows = []
    for source in sources:
        target = snapshot / source.name
        shutil.copyfile(source, target)
        source_rows.append({"path": str(source), "snapshot_path": str(target), "sha256": digest(source)})
    config_snapshot = output / "detector_config.json"
    shutil.copyfile(detector_config, config_snapshot)
    model_checks = []
    for field in ("weights", "training_receipt"):
        actual = digest(config[field])
        model_checks.append({"role": field, "path": config[field], "expected_sha256": config[field+"_sha256"],
            "actual_sha256": actual, "status": "PASS" if actual == config[field+"_sha256"] else "FAIL"})
    if any(row["status"] != "PASS" for row in model_checks):
        raise ValueError("Frozen model or receipt hash mismatch")
    created = datetime.now(timezone.utc)
    session = "t1_vlx_latency_"+created.strftime("%Y%m%dT%H%M%S%fZ")
    policy = PolicyConfig()
    report = {"kind": "T1_SAVED_RGB_VLX_TRANSPORT_LATENCY_DIAGNOSTIC", "status": "RUNNING",
        "created_utc": created.isoformat(), "session_id": session, "rgb_path": str(rgb), "rgb_sha256": image_sha,
        "detector_config_path": str(detector_config), "detector_config_sha256": config_sha,
        "description_sha256": hashlib.sha256(DESCRIPTION.encode()).hexdigest(),
        "description": DESCRIPTION, "planned_requests": requests, "rows": [], "sources": source_rows,
        "model_artifact_checks": model_checks, "policy_response_timeout_s": policy.response_timeout_s,
        "policy_semantic_source_max_age_s": policy.semantic_max_age_s,
        "client_request_timeout_s": 3, "client_max_attempts": 1,
        "request_mode": "image_text", "proposals_transmitted": False, "new_flights": 0,
        "formal_efficacy_results": False, "population_latency_estimate": False,
        "time_scope": "new host receipt and YOLO preparation from a saved RGB; original camera exposure age is excluded",
        "request_timeout_scope": "requests transport timeout; not a guaranteed overall process wall-clock bound",
        "policy_scope": "direct backend completion diagnostic; does not mint a current-frame semantic certificate",
        "ai_disclosure": "AI-assisted diagnostic script; numeric results computed from receipts and author should review interpretation"}
    save(output / "report.json", report)
    observer = None
    sink = DiscardOutput()
    # Factory/model libraries may print; discard rather than capture and serialize.
    with redirect_stdout(sink), redirect_stderr(sink):
        try:
            observer = create_observer(detector_config)
            client = observer.backend.vlx_client
            if client.timeout != 3 or client.max_retries != 1:
                raise ValueError("Diagnostic requires factory timeout 3 and one attempt")
            model = {key: value for key, value in observer.model_receipt.items() if key in MODEL_FIELDS}
            report["model_binding"] = model
            warmup = observer.backend.warmup(rgb)
            report["warmup"] = {key: warmup[key] for key in (
                "status", "wall_seconds", "image_sha256", "semantic_api_called", "trial_count") if key in warmup}
            for index in range(requests):
                report["rows"].append(one_request(observer.backend, rgb, image_sha, session, index))
                save(output / "report.json", report)
            report["status"] = "DIAGNOSTIC_COMPLETED"
            report["client_usage_numeric_counters"] = usage(client)
        except Exception as error:
            report.update(status="LOCAL_DIAGNOSTIC_FAILURE", error_category=safe_error_category(error))
        finally:
            if observer is not None:
                try:
                    observer.close()
                except Exception as error:
                    report["close_error_category"] = safe_error_category(error)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {"attempt_rows": len(report["rows"]),
        "backend_replies_returned": sum(row.get("backend_reply_returned") is True for row in report["rows"]),
        "response_timeout_2s_met_count": sum(row.get("response_timeout_2s_met") is True for row in report["rows"]),
        "semantic_source_age_2s_met_count": sum(row.get("semantic_source_age_2s_met") is True for row in report["rows"]),
        "unique_source_association_count": sum(row.get("association_status") == "UNIQUE_SOURCE_YOLO_ASSOCIATION" for row in report["rows"])}
    save(output / "report.json", report)
    rows = ["# VLX 小样本传输时延诊断", "", "同一保存 RGB、当前冻结 YOLO 权重、顺序调用，最多 3 次。无需启动仿真。", "",
        "| 请求 | 返回状态 | 请求耗时（s） | 2 s 响应预算 | 返回框数 | 本地源框关联 | 错误类别 |", "|---|---|---:|---|---:|---|---|"]
    for row in report["rows"]:
        wall = f"{row['request_wall_s']:.3f}" if number(row.get("request_wall_s")) else "未知"
        rows.append(f"| {row['request_index']} | {row['status']} | {wall} | {row.get('response_timeout_2s_met', '未知')} | {row.get('bbox_count')} | {row.get('association_status', '未知')} | {row.get('error_category')} |")
    rows += ["", "返回绑定由本地客户端建立，未声称服务回显；当前诊断不会产生飞行身份凭证。新主机时间仅用于保存图像重放，不能视作新相机观测。", "",
        "这些少量调用用于定位响应预算、源观测期限及解析关联的问题，不能据此估计总体服务时延或判定所有网络故障。API 传输 timeout 为 3 s、尝试 1 次，政策响应和源观测期限保持 2 s。", "",
        "未保存端点、密钥、响应原文或异常消息。报告由 AI 辅助生成，解释需由作者结合原始数值审核。", ""]
    summary = output / "summary.md"
    summary.write_text("\n".join(rows), encoding="utf-8")
    receipt = {"kind": "T1_VLX_LATENCY_DIAGNOSTIC_ARTIFACT_RECEIPT", "status": report["status"],
        "sources": source_rows, "input_image_sha256": image_sha, "input_config_sha256": config_sha,
        "model_artifact_checks": model_checks,
        "artifacts": [{"path": str(path), "sha256": digest(path)} for path in (
            output / "report.json", summary, config_snapshot)], "new_flights": 0,
        "requests_planned": requests, "attempt_rows": len(report["rows"]), "formal_efficacy_results": False}
    save(output / "artifact_receipt.json", receipt)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    try:
        result = profile(args.rgb, args.detector_config, args.output, args.requests)
        print(json.dumps({"status": result["status"], "summary": result["summary"],
            "output": str(args.output.resolve())}, ensure_ascii=False))
    except Exception as error:
        # Never let a raw transport/configuration exception reach stdout/stderr.
        print(json.dumps({"status": "DIAGNOSTIC_SETUP_FAILED", "error_category": safe_error_category(error)}, ensure_ascii=False))
        raise SystemExit(1)

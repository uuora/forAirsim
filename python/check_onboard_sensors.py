"""Read-only, repeated sensor acceptance; no arming, flight or world changes."""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import airsim

from onboard_sensors import OnboardSensorMonitor, SENSORS

ROOT = Path(__file__).resolve().parents[1]


def assess_sequence(samples, vehicles, max_stale_s):
    """A successful RPC alone does not prove that a sensor is updating."""
    failed, summary = [], {}
    for vehicle in vehicles:
        summary[vehicle] = {}
        for kind, _, _ in SENSORS:
            items = [sample[vehicle]["sensors"][kind] for sample in samples]
            timestamps = [x["timestamp_ns"] for x in items if x["timestamp_ns"] is not None]
            reasons, limitations = [], []
            # Do not relabel negative measurements valid. Known near-zero
            # simulator noise is a measurement limitation, not a missing RPC.
            def expected_negative_range(item):
                data = item.get("data", {})
                return (kind.startswith("distance_")
                        and data.get("range_status") == "negative_invalid"
                        and -0.6 <= data.get("distance_m", -math.inf) < 0
                        and math.isfinite(data.get("max_distance_m", math.nan))
                        and 0 <= data.get("min_distance_m", -1) < data.get("max_distance_m", -1))
            if any(not x["valid"] and not expected_negative_range(x) for x in items):
                reasons.append("invalid_payload_or_rpc_error")
            unusable = sum(not x["valid"] or not x.get("data", {}).get("in_range", False)
                           for x in items) if kind.startswith("distance_") else 0
            if unusable:
                limitations.append("unusable_range_samples; retain raw values, exclude from ranging")
            if any(not x["fresh"] for x in items):
                reasons.append("stale_or_invalid_timestamp")
            if not any(b > a for a, b in zip(timestamps, timestamps[1:])):
                reasons.append("no_timestamp_progress_observed")
            if any(b < a for a, b in zip(timestamps, timestamps[1:])):
                reasons.append("timestamp_regression")
            successful = [x for x in items if "host_response_monotonic_s" in x]
            elapsed = (successful[-1]["host_response_monotonic_s"] -
                       successful[0]["host_response_monotonic_s"]) if len(successful) > 1 else 0
            if elapsed <= max_stale_s:
                reasons.append("observation_window_too_short")
            summary[vehicle][kind] = {
                "status": "FAIL" if reasons else "LIMITED" if limitations else "PASS",
                "reasons": reasons, "limitations": limitations,
                "unusable_range_samples": unusable,
                "samples": len(items), "valid": sum(x["valid"] for x in items),
                "unique_timestamps": len(set(timestamps)), "observed_wall_span_s": elapsed,
                "saturated_samples": sum(x.get("data", {}).get("saturated", False) for x in items),
                "below_minimum_samples": sum(x.get("data", {}).get("below_minimum", False) for x in items),
                "in_range_samples": sum(x.get("data", {}).get("in_range", False) for x in items),
                "last_data": items[-1].get("data"),
            }
            if reasons:
                failed.append({"vehicle": vehicle, "sensor": kind, "reasons": reasons})
    return failed, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/maritime_harbor_settings.json")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-stale", type=float, default=2.5)
    parser.add_argument("--cameras", action="store_true", help="Also save RGB/depth samples and camera geometry audit")
    args = parser.parse_args()
    if (not all(math.isfinite(x) for x in (args.duration, args.interval, args.max_stale))
            or args.max_stale <= 0 or args.duration <= args.max_stale
            or not 0 < args.interval < args.duration):
        parser.error("finite duration > max-stale > 0 and 0 < interval < duration required")
    settings_bytes = args.settings.read_bytes()
    settings = json.loads(settings_bytes)
    vehicles = list(settings["Vehicles"])
    port = settings.get("ApiServerPort", 41451)
    folder = ROOT / "logs" / datetime.now().strftime("sensors_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    (folder / "settings.json").write_bytes(settings_bytes)
    report = {"status": "RUNNING", "settings_path": str(args.settings.resolve()),
              "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(), "port": port,
              "vehicles": vehicles, "flight_commands_issued": False,
              "policy": {"role": "advisory_only", "max_stale_s": args.max_stale,
                         "freshness_scope": "timestamp_progress_watchdog_not_absolute_age",
                         "range_health": "saturation/below-minimum are flagged; not usable ranges",
                         "sample_rate_scope": "RPC polling; not native sensor output frequency"}}
    try:
        client = airsim.MultirotorClient(ip="127.0.0.1", port=port, timeout_value=5)
        if not client.ping():
            raise RuntimeError("AirSim RPC ping failed")
        live = json.loads(client.getSettingsString())
        if live.get("Vehicles") != settings["Vehicles"]:
            raise RuntimeError("Live vehicle settings differ; restart the scene to load the supplied settings")
        if not set(vehicles).issubset(client.listVehicles()):
            raise RuntimeError("Configured vehicles are missing from live scene")
        if client.simIsPause():
            raise RuntimeError("Simulation is paused; a timestamp-progress test needs a running clock")
        monitor = OnboardSensorMonitor(client, max_stale_s=args.max_stale)
        samples, started = [], time.monotonic()
        with (folder / "sensor_samples.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                sample = monitor.read_all(vehicles)
                samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                if time.monotonic() - started >= args.duration:
                    break
                time.sleep(args.interval)
        failed, summary = assess_sequence(samples, vehicles, args.max_stale)
        report.update(failed_sensors=failed, sensor_summary=summary,
                      polls=len(samples), duration_observed_s=time.monotonic() - started)
        if args.cameras:
            from sensor_camera_audit import audit_cameras
            report["camera_audit"] = audit_cameras(client, settings, folder)
        limited = any(s["status"] == "LIMITED" for v in summary.values() for s in v.values())
        report["status"] = ("FAIL" if failed or report.get("camera_audit", {}).get("status") == "FAIL"
                            else "PASS_WITH_LIMITATIONS" if limited else "PASS")
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        output = folder / "sensor_report.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "port": port,
                          "polls": report.get("polls"), "failed_sensors": report.get("failed_sensors"),
                          "error": report.get("error"), "report": str(output)}, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

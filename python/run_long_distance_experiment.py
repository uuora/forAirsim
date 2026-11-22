"""Run the paired long-distance routes through the normal AirSim panel."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from dual_target_control.simulator import ROOT, port_open


ROUTES = {
    "straight_36": ("ordinary", "toppquad-fixed", "toppquad"),
    "straight_72": ("toppquad-fixed", "toppquad", "ordinary"),
    "turn_90": ("toppquad", "ordinary", "toppquad-fixed"),
    "turn_135": ("ordinary", "toppquad", "toppquad-fixed"),
    "climb": ("toppquad", "toppquad-fixed", "ordinary"),
    "descent": ("toppquad-fixed", "ordinary", "toppquad"),
}


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_selected(routes):
    preset_file = ROOT / "configs/long_distance_routes.json"
    flight_config = ROOT / "configs/fast_route.json"
    airsim = read_json(ROOT / "configs/dual_target_control.json")["airsim"]
    directory = ROOT/"reports"/f"long_distance_run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True)
    schedule = [dict(route=route, method=method, state="PENDING") for route in routes for method in ROUTES[route]]
    design = dict(route_presets=read_json(preset_file), flight_config=read_json(flight_config),
                  preset_sha256=hashlib.sha256(preset_file.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(flight_config.read_bytes()).hexdigest(),
                  geometry="hermite", policy="pass", fresh_simulator_per_flight=True,
                  order=[dict(row) for row in schedule])
    write_json(directory/"design.json", design)
    results_path = directory/"progress.json"
    write_json(results_path, schedule)
    print(directory, flush=True)
    env = dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, row in enumerate(schedule, 1):
        tag = f"{row['route']}/{row['method']}"
        if port_open(airsim["host"], airsim["port"]):
            row.update(state="BLOCKED", error="Existing AirSim instance. The campaign starts a fresh instance per flight.")
            write_json(results_path, schedule)
            print(f"{tag}: {row['error']}", flush=True)
            break
        print(f"{index}/{len(schedule)} {tag}", flush=True)
        row["state"] = "RUNNING"
        write_json(results_path, schedule)
        command = [sys.executable, "-u", str(ROOT/"python/fast_panel.py"),
                   "--route-preset", row["route"], "--verify-flights", f"pass/hermite/{row['method']}"]
        log = directory/f"{index:02d}_{row['route']}_{row['method']}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                                    stdout=subprocess.PIPE, stderr=stream, check=False)
            stream.write(result.stdout)
        row["panel_exit_code"] = result.returncode
        try:
            batch = Path(result.stdout.strip().splitlines()[-1]).resolve()
            if batch.parent != (ROOT/"reports").resolve():
                raise ValueError("Panel did not return a local report directory")
            receipt = read_json(batch/"results.json")
            audit = read_json(batch/"audit.json")
            if receipt["route_preset"] != row["route"] or audit["requested"] != 1:
                raise ValueError("Batch does not match the selected route")
            row["batch"] = batch.name
            flight, = audit["flights"]
            if flight["arm"] != row["method"] or flight["geometry"] != "hermite":
                raise ValueError("Flight does not match the selected method")
            row.update(state="RECORDED", run=flight["run"], passed=flight["passed"],
                       outcome=flight["outcome"], safe_landing=flight["safe_landing"],
                       error=flight.get("error"), failed_checks=flight.get("failed_checks", []))
        except (IndexError, OSError, KeyError, TypeError, ValueError) as exc:
            row.update(state="ERROR", error=str(exc))
        write_json(results_path, schedule)
        print(f"{tag}: {row['state']} pass={row.get('passed')} landed={row.get('safe_landing')}", flush=True)
        if not row.get("safe_landing"):
            break
    return int(not all(row.get("passed") for row in schedule))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airsim", action="store_true", required=True, help="Run the selected AirSim flights")
    parser.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=list(ROUTES))
    args = parser.parse_args()
    if len(set(args.routes)) != len(args.routes):
        parser.error("Select each route at most once")
    raise SystemExit(run_selected(args.routes))

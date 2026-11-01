"""Read-only readiness receipt for the control-simulation research plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]


def probe():
    locations = {
        "unreal_editor": Path("D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe"),
        "blocks_project": ROOT / "external/AirSim/Unreal/Environments/Blocks/Blocks.uproject",
        "airsim_plugin": ROOT / "external/AirSim/Unreal/Environments/Blocks/Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll",
        "timed_tracker": ROOT / "python/trajectory_research/timed_tracking.py",
        "curve_parameters": ROOT / "python/trajectory_research/parameterizations.py",
    }
    modules = {name: importlib.util.find_spec(name) is not None
               for name in ("numpy", "matplotlib", "torch", "airsim")}
    connected = False
    error = None
    try:
        with socket.create_connection(("127.0.0.1", 41451), timeout=.5):
            connected = True
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    ping = None
    if connected and modules["airsim"]:
        try:
            import airsim
            ping = bool(airsim.MultirotorClient(ip="127.0.0.1", port=41451,
                                               timeout_value=2).ping())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    simplified_ready = modules["numpy"] and modules["matplotlib"] \
        and locations["timed_tracker"].is_file() and locations["curve_parameters"].is_file()
    return dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        operation="READ_ONLY_AVAILABILITY_CHECK_NO_FLIGHT_COMMANDS",
        python_executable=sys.executable, python_version=platform.python_version(),
        platform=platform.platform(), modules=modules,
        paths={key: dict(path=str(path), exists=path.is_file()) for key, path in locations.items()},
        simplified_backend_ready=bool(simplified_ready),
        airsim_endpoint="127.0.0.1:41451", tcp_connected=connected,
        airsim_ping=ping, connection_error=error,
        airsim_execution_performed=False,
        interpretation=("AirSim RPC ping succeeded; research execution/reset protocol still requires separate validation."
                        if ping else "AirSim is not connected at the declared endpoint. Installed files alone do not establish a running simulation."),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit("Refusing to overwrite an existing environment receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = probe()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))

"""Windows double-click launcher without PowerShell execution-policy dependency."""
import argparse
import csv
import io
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR = Path("D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe")
PROJECT = ROOT / "external/AirSim/Unreal/Environments/Blocks/Blocks.uproject"


def rpc_ready():
    try:
        with socket.create_connection(("127.0.0.1", 41451), timeout=0.5):
            return True
    except OSError:
        return False


def editor_running():
    result = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq UE4Editor.exe", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, errors="replace", check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return any(row and row[0].lower() == "ue4editor.exe" for row in csv.reader(io.StringIO(result.stdout)))


def run_script(name, *args):
    subprocess.run([sys.executable, str(ROOT / "python" / name), *args], cwd=ROOT, check=True)


def start_camera_monitor():
    """Keep a separate live camera window open while a mission is flying."""
    return subprocess.Popen([sys.executable, str(ROOT / "python" / "live_camera_view.py")],
                            cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("a_hit", "fallback", "both", "vision", "visual_range", "visual_contact", "visual_fallback"), default="fallback")
    parser.add_argument("--attempt-mode", choices=("fault_injection", "observation"),
                        default="fault_injection",
                        help="For visual_fallback: retain the injected A miss or assess it from observations")
    parser.add_argument("--check", action="store_true", help="Check launcher dependencies without starting or flying")
    args = parser.parse_args()
    for path in (EDITOR, PROJECT, ROOT / "python/setup_scene.py", ROOT / "python/run_balloon_mission.py",
                 ROOT / "python/summarize_experiments.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    subprocess.run([sys.executable, "-c", "import airsim, cv2, numpy"], check=True, cwd=ROOT)
    ready = rpc_ready()
    running = editor_running()
    print(f"Launcher ready. Python: {sys.executable}", flush=True)
    print(f"AirSim port ready: {ready}; UE process running: {running}", flush=True)
    if args.check:
        print("CHECK PASS. No simulation changes were made.")
        return
    if not ready:
        if not running:
            # The simulator is an interactive window for the user to watch.
            subprocess.Popen([str(EDITOR), str(PROJECT), "-game", "-windowed", "-ResX=1280", "-ResY=720"], cwd=ROOT)
        else:
            print("Waiting for existing UE. If the editor is open, press Play in Blocks.", flush=True)
        deadline = time.monotonic() + 120
        while not rpc_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError("AirSim port 41451 is not ready. Check the Blocks window.")
            time.sleep(1)
    scenarios = ("a_hit", "fallback") if args.scenario == "both" else (args.scenario,)
    monitor = None
    if args.scenario in ("a_hit", "fallback", "both"):
        monitor = start_camera_monitor()
        print("Live camera monitor opened. Close it with Q/Esc; the mission continues.", flush=True)
    for scenario in scenarios:
        print(f"Preparing {scenario}. Switch to Blocks and press M for overview.", flush=True)
        run_script("setup_scene.py")
        if scenario == 'visual_range':
            run_script('run_visual_alignment.py', '--range')
            print('Visual range test completed. Both vehicles landed; simulation paused.')
            return
        if scenario == 'visual_contact':
            run_script('run_visual_alignment.py', '--range', '--contact')
            print('Visual handoff contact test completed. Both vehicles landed; simulation paused.')
            return
        if scenario == 'visual_fallback':
            run_script('run_visual_fallback.py', '--attempt-mode', args.attempt_mode)
            print('Visual fallback test completed. Both vehicles landed; simulation paused.')
            return
        if scenario == "vision":
            run_script("run_approach_demo.py")
            run_script("check_vision.py", "--validate", "--show")
            print("Vision completed. Drones remain airborne with simulation paused for inspection.")
            return
        run_script("run_balloon_mission.py", "--scenario", scenario)
    run_script("summarize_experiments.py")
    print("Completed. Both drones landed and disarmed; simulation paused.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        print(f"ERROR: {exc}\nIf this says AirSim port 41451, start Blocks and press Play, then rerun the launcher.\nCheck the message above and the project logs folder.", file=sys.stderr)
        sys.exit(1)

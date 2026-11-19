"""Paths and process options shared with geometric AirSim execution."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def workspace_path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def hidden_process_options():
    if os.name != 'nt':
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {'startupinfo': startup, 'creationflags': subprocess.CREATE_NO_WINDOW}

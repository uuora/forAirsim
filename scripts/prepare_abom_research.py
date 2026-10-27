"""Fetch/check the exact upstream ABOM network used by offline research.

Only abom_net.py is obtained from the declared upstream commit. It remains in
the ignored tmp directory; this script does not vendor third-party source.
The Git blob hash is checked before writing and again when reading an existing
file. Existing mismatched files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "79a2baec700ce0abc6cf7d4cba1e897989d80516"
EXPECTED_GIT_BLOB = "5959aa7e4aef13c6b128e7c16e51d8a23205d030"
SOURCE_URL = f"https://raw.githubusercontent.com/xiaofangxd/ABOM/{UPSTREAM_COMMIT}/abom_net.py"
DEFAULT_DESTINATION = ROOT / "tmp" / "abom_deploy_probe" / "upstream"
MAX_BYTES = 2 * 1024 * 1024


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def verify(data: bytes) -> str:
    actual = git_blob(data)
    if actual != EXPECTED_GIT_BLOB:
        raise ValueError(f"ABOM network Git blob mismatch: expected {EXPECTED_GIT_BLOB}, got {actual}")
    return actual


def checked_file(path: Path) -> dict:
    data = path.read_bytes()
    blob = verify(data)
    return dict(path=str(path), bytes=len(data), git_blob=blob,
                sha256=hashlib.sha256(data).hexdigest(), upstream_commit=UPSTREAM_COMMIT)


def download() -> bytes:
    last_error = None
    for attempt in range(3):
        request = Request(SOURCE_URL, headers={"User-Agent": "forAirsim-ABOM-research-reproduction/1"})
        try:
            with urlopen(request, timeout=20) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("ABOM source response exceeds the 2 MiB bound")
            verify(data)  # Reject unexpected content before creating the target file.
            return data
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(.5 * (attempt + 1))
    raise RuntimeError(f"ABOM download failed after 3 bounded attempts: {last_error}")


def prepare(destination: Path, *, check_only=False) -> dict:
    destination = Path(destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    target = destination / "abom_net.py"
    if target.exists():
        result = checked_file(target)
        result["status"] = "verified_existing"
        return result
    if check_only:
        raise FileNotFoundError(f"No ABOM network to check: {target}; rerun without --check-only to download")
    data = download()
    destination.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents an intervening file from being replaced.
    try:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        result = checked_file(target)
        result["status"] = "verified_existing_after_concurrent_creation"
        return result
    result = checked_file(target)
    result.update(status="downloaded_and_verified", source_url=SOURCE_URL)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help="directory for abom_net.py; relative paths are relative to the repository")
    parser.add_argument("--check-only", action="store_true", help="verify an existing file without network access")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.destination, check_only=args.check_only), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ABOM preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

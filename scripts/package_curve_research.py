"""Package an explicit research-evidence allowlist into bounded, hashed ZIPs.

No workspace-wide scan, credentials, source presentations, downloaded papers or
third-party network implementation is included. The manifest is an inventory,
not an attestation that the experiments were independently executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ("search_pilot", "curve_search", "uncertainty_nominal", "uncertainty_margin",
            "ellipsoid_nominal", "ellipsoid_nominal_final", "ellipsoid_isotropic", "ellipsoid_directional", "geometry")
CURRENT = ("main", "budget_extension")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gather(round_name):
    base = ROOT / "reports" / ("hermite_disturbance_20261027" if round_name == "previous"
                               else "hermite_parameter_perturbation_20261027")
    campaigns = PREVIOUS if round_name == "previous" else CURRENT
    files = []
    for name in campaigns:
        folder = base / name
        if not folder.is_dir():
            raise ValueError(f"Required evidence folder missing: {folder}")
        if name != "geometry" and name != "ellipsoid_nominal" and not (folder / "summary.json").exists():
            raise ValueError(f"Campaign incomplete: {folder}")
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".jsonl", ".py")
                     and "__pycache__" not in p.parts)
    for name in (("integrity_audit.json", "integrity_audit.md", "paired_profile_comparisons.json") if round_name == "previous"
                 else ("decoder_equivalence.json", "energy_identity.json", "method_review.md")):
        path = base / name
        if path.exists():
            files.append(path)
    # No implicit additions: temporary ingestion material and previous delivery
    # bundles can contain redundant or user-supplied copyrighted source assets.
    return sorted(set(files)), base


def package(round_name, destination, *, max_mib=24.):
    if not 1 <= max_mib <= 90:
        raise ValueError("max_mib must be in [1,90]")
    files, base = gather(round_name)
    destination.mkdir(parents=True, exist_ok=True)
    entries, groups, group, compressed = [], [], [], 0
    for path in files:
        data = path.read_bytes()
        entry = dict(path=path.relative_to(ROOT).as_posix(), size=len(data), sha256=sha(data))
        estimate = len(zlib.compress(data, 9)) + len(entry["path"].encode()) * 2 + 256
        if estimate > max_mib * 1024 ** 2:
            raise ValueError(f"Single file exceeds archive size target: {path}")
        if group and compressed + estimate > max_mib * 1024 ** 2 - 1024 * 1024:
            groups.append(group)
            group, compressed = [], 0
        entries.append(entry)
        group.append(entry)
        compressed += estimate
    if group:
        groups.append(group)
    archives = []
    stem = "previous_round_evidence" if round_name == "previous" else "new_round_evidence"
    for index, group in enumerate(groups, 1):
        filename = stem + ("" if len(groups) == 1 else f"_part{index:02d}") + ".zip"
        path = destination / filename
        archive_manifest = dict(schema="curve-evidence-archive/1", round=round_name,
                                part=index, parts=len(groups), files=group)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry in group:
                archive.write(ROOT / entry["path"], entry["path"])
            archive.writestr("ARCHIVE_MANIFEST.json", json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n")
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Archive CRC failure: {bad}")
            for entry in group:
                if sha(archive.read(entry["path"])) != entry["sha256"]:
                    raise ValueError("Archive payload hash mismatch")
        data = path.read_bytes()
        if len(data) > max_mib * 1024 ** 2:
            raise ValueError("Compressed archive exceeds declared part size target")
        archives.append(dict(file=filename, size=len(data), sha256=sha(data), files=len(group)))
    manifest = dict(schema="curve-evidence-inventory/1", round=round_name, archives=archives,
                    files=len(entries), uncompressed_bytes=sum(x["size"] for x in entries),
                    source_directory=base.relative_to(ROOT).as_posix(),
                    exclusions=["user source presentation/intake assets", "downloaded papers", "third-party ABOM source",
                                "environment files and credentials", "render/build caches", "smoke/development runs"],
                    inventory=entries)
    (destination / (stem + "_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("round", "archives", "files", "uncompressed_bytes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", choices=("previous", "current"))
    parser.add_argument("--destination", type=Path, default=ROOT / "experiments/hermite_perturbation_20261027")
    parser.add_argument("--max-mib", type=float, default=24.)
    args = parser.parse_args()
    package(args.round, args.destination.resolve(), max_mib=args.max_mib)

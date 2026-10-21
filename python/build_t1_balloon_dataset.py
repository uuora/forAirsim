"""Build a preregistered, session-grouped controlled balloon YOLO dataset.

Only archived RGB is exported to images. Instance masks supply class-0 visible
box supervision; they are never input to a detector, VLX or a control policy.
All provenance, scene and split checks finish before the output is created.
This small same-map/same-material adaptation set is not formal thesis efficacy
or evidence of appearance/map/real-world generalization. AI assisted this code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import uuid

import cv2
import numpy as np

from export_t1_balloon_dataset import validate_episode_split, yolo_label
from t1_ground_perception_probe import visible_mask


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
YOLO_SPLIT = {"train": "train", "validation": "val", "test": "test"}
REQUIRED_SOURCES = {"t1_ground_perception_probe.py", "setup_scene.py", "camera_geometry.py"}
INSTANCE_IDS = {"T1ProbeTarget": 42, "T1ProbeDistractor": 43, "T1ProbeOccluder": 44}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetBuildError(ValueError):
    """A rejected build with a separate, non-overwriting diagnostic receipt."""

    def __init__(self, message, receipt_path=None):
        super().__init__(message)
        self.receipt_path = receipt_path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes_sha(data):
    return hashlib.sha256(data).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value, base, name):
    _require(isinstance(value, str) and bool(value.strip()), name + " must be filled before building")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _split(value):
    result = "validation" if value == "val" else value
    _require(result in SPLITS, "Expected an explicit train/validation/test split")
    return result


def _identifier(value, name):
    _require(isinstance(value, str) and bool(SAFE_NAME.fullmatch(value)), "Unsafe or absent " + name)
    _require(value not in (".", ".."), "Unsafe " + name)
    return value


def _positive_int(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             name + " must be a positive integer")
    return value


def _hash(value, name):
    _require(isinstance(value, str) and bool(SHA256.fullmatch(value)), "Invalid " + name)
    return value


def _utc_identity(value):
    """Canonical UTC start instant without dropping .NET's seventh fraction digit."""
    _require(isinstance(value, str), "Missing UE start UTC")
    match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d+))?(?:Z|\+00:00)", value)
    _require(match is not None, "Invalid UE start UTC format")
    parsed = _parse_instant(value)
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             "UE start must be an explicit UTC instant")
    fraction = (match.group(1) or "").rstrip("0")
    return parsed.replace(microsecond=0).isoformat() + ("|" + fraction if fraction else "")


def _parse_instant(value):
    # Python 3.8 rejects .NET's seven fraction digits. Retain the original full
    # fraction separately in the process key; this parse is only for date checks.
    compatible = re.sub(r"\.(\d{6})\d+(?=Z|[+-]\d{2}:\d{2}$)", r".\1", value)
    return datetime.fromisoformat(compatible.replace("Z", "+00:00"))


def _process(manifest):
    return (_positive_int(manifest.get("engine_process_id"), "UE PID"),
            _utc_identity(manifest.get("engine_started_utc")))


def _image(path):
    array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    _require(array is not None and array.ndim == 3 and array.shape[2] == 3,
             "RGB/segmentation PNG could not be decoded")
    return array


def _pixel_sha(array):
    # Also detect exact same decoded RGB saved using different PNG compression.
    dimensions = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    return _bytes_sha(dimensions + b"\0" + np.ascontiguousarray(array).tobytes())


def _ground(states, where):
    _require(isinstance(states, dict) and set(states) == {"DroneA", "DroneB"},
             where + " needs both grounded vehicles")
    for state in states.values():
        speed = state.get("speed_m_s")
        _require(state.get("landed") is True and state.get("api_control_enabled") is False,
                 where + " is not grounded/released")
        _require(isinstance(speed, (int, float)) and not isinstance(speed, bool)
                 and math.isfinite(speed) and 0 <= speed <= .02,
                 where + " is not stationary")


def _palette(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        number, value = line.split("\t", 1)
        color = json.loads(value)
        _require(isinstance(color, list) and len(color) == 3
                 and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 255 for x in color),
                 "Invalid independently archived segmentation palette")
        _require(number not in result, "Duplicate segmentation palette ID")
        result[number] = color
    _require(all(str(i) in result for i in (0, 42, 43, 44)), "Required segmentation palette IDs absent")
    return {str(i): result[str(i)] for i in (0, 42, 43, 44)}


def _label_text(labels):
    return "".join(" ".join([str(row[0])] + [f"{value:.9f}" for value in row[1:]]) + "\n"
                   for row in labels)


def _scene(case, observed):
    """Recompute collector's strict intended-instance rule from actual masks."""
    _require("distractor_world_ned_m" in case or observed["distractor"]["visible_pixel_count"] == 0,
             "Inactive distractor is visible")
    _require("occluder_world_ned_m" in case or observed["occluder"]["visible_pixel_count"] == 0,
             "Inactive occluder is visible")
    if "occluder_world_ned_m" in case:
        _require(observed["target"]["visible_pixel_count"] == 0, "Full target occlusion was not achieved")
    else:
        _require(observed["target"]["visible_pixel_count"] >= 500, "Intended visible target has <500 pixels")
    if "distractor_world_ned_m" in case:
        _require(observed["distractor"]["visible_pixel_count"] >= 500,
                 "Intended visible distractor has <500 pixels")


def _exclusions(plan, base):
    """Keep the already consumed development sessions/images out of the new set."""
    dataset_paths = [ROOT / "datasets/t1_balloon_development_20261004/manifest.json",
                     ROOT / "datasets/t1_balloon_development_v1p1_20261004/manifest.json"]
    dataset_paths += [_path(value, base, "excluded_development_manifest")
                      for value in plan.get("excluded_development_manifests", [])]
    campaign_paths = [ROOT / "logs" / name / "manifest.json" for name in (
        "t1_ground_camera_campaign_20261004_000437_5286608",
        "t1_ground_camera_campaign_20261004_000731_5735269",
        "t1_ground_camera_campaign_20261004_003218_3495496",
        "t1_ground_camera_campaign_20261004_003840_1015193")]
    excluded = {"reports": set(), "processes": set(), "sessions": set(),
                "rgb_hashes": set(), "pixel_hashes": set(), "receipts": []}
    for path in dict.fromkeys(dataset_paths):
        if not path.exists():
            # Built-in historical artifacts need not exist in isolated unit fixtures.
            _require(path in dataset_paths[:2], "Explicit exclusion manifest is missing")
            continue
        data = _load(path)
        excluded["receipts"].append({"kind": "development_dataset", "path": str(path), "sha256": _sha(path)})
        source = data.get("source_report")
        if source:
            source_path = _path(source, path.parent, "development source_report")
            excluded["reports"].add(str(source_path))
            launcher = source_path.parent.parent / "manifest.json"
            if launcher.exists():
                campaign_paths.append(launcher)
        for row in data.get("rows", []):
            if row.get("episode_id"):
                excluded["sessions"].add(row["episode_id"])
            if row.get("rgb_sha256"):
                excluded["rgb_hashes"].add(row["rgb_sha256"])
            if row.get("image"):
                image = _path(row["image"], path.parent, "development image")
                _require(image.is_file() and _sha(image) == row["rgb_sha256"],
                         "Excluded development image failed provenance check")
                excluded["pixel_hashes"].add(_pixel_sha(_image(image)))
    for path in dict.fromkeys(campaign_paths):
        if not path.exists():
            continue
        manifest = _load(path)
        excluded["receipts"].append({"kind": "development_campaign", "path": str(path), "sha256": _sha(path)})
        if manifest.get("engine_process_id") and manifest.get("engine_started_utc"):
            excluded["processes"].add(_process(manifest))
        source = manifest.get("report")
        if source:
            report_path = _path(source, path.parent, "development report")
            excluded["reports"].add(str(report_path))
            if report_path.exists():
                for frame in _load(report_path).get("frames", []):
                    excluded["sessions"].add(frame["session_id"])
                    excluded["rgb_hashes"].add(frame["media"]["rgb"]["sha256"])
    return excluded


def _validate_episode(row, plan, base, palette_path, palette, excluded, plan_path):
    episode_id = _identifier(row.get("episode_id"), "episode_id")
    split = _split(row.get("split"))
    seed = row.get("acquisition_seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
             "A preregistered acquisition_seed integer is required")
    protocol_path = _path(row.get("protocol_path"), base, "protocol_path")
    manifest_path = _path(row.get("manifest_path"), base, "manifest_path")
    report_path = _path(row.get("report_path"), base, "report_path")
    protocol_hash = _hash(row.get("protocol_sha256"), "preregistered protocol SHA256")
    _require(_sha(protocol_path) == protocol_hash, "Preregistered protocol hash mismatch")
    protocol = _load(protocol_path)
    manifest = _load(manifest_path)
    report = _load(report_path)
    _require(manifest.get("status") == "PASS" and manifest.get("engine_closed_after_safe_completion") is True,
             "Launcher PASS and confirmed owned engine closure required")
    _require(manifest.get("flight_requests") is False and manifest.get("formal_efficacy_results") is False,
             "Only ground detector-data launcher evidence is accepted")
    _require(manifest.get("episode_id") == episode_id and _split(manifest.get("split")) == split
             and manifest.get("acquisition_seed") == seed, "Launcher episode/split/seed disagrees with preregistration")
    process = _process(manifest)
    if plan.get("created_utc"):
        frozen_at = _parse_instant(plan["created_utc"])
        started_at = _parse_instant(manifest["engine_started_utc"])
        _require(frozen_at.tzinfo is not None and started_at >= frozen_at,
                 "UE session predates the declared plan freeze")
    _require(process not in excluded["processes"], "UE process was already used for development")
    _require(str(report_path) not in excluded["reports"], "Report was already used for development")
    _require(_path(manifest.get("report"), manifest_path.parent, "launcher report") == report_path
             and _sha(report_path) == _hash(manifest.get("report_sha256"), "launcher report SHA256"),
             "Launcher report hash/path mismatch")
    for owner in (manifest, report):
        _require(_path(owner.get("protocol"), manifest_path.parent if owner is manifest else report_path.parent,
                       "protocol receipt") == protocol_path and owner.get("protocol_sha256") == protocol_hash,
                 "Protocol receipt hash/path mismatch")
    _require(protocol.get("acquisition_seed") == seed and _split(protocol.get("split")) == split,
             "Protocol seed/split disagrees with preregistration")
    _require(protocol.get("formal_efficacy_results") is False, "Protocol incorrectly claims formal efficacy")
    _require(report.get("configuration") == protocol, "Report configuration differs from frozen protocol")
    _require(report.get("status") == "PASS" and report.get("safe_ground_closure") is True
             and report.get("paused_after") is True, "Report PASS with safe paused ground closure required")
    _require(report.get("flight_commands_issued") is False and report.get("api_control_acquired") is False
             and report.get("independent_UE_sessions") == 1 and report.get("formal_efficacy_results") is False,
             "Report is not a single ground-only development session")
    _require(report.get("evaluation_segmentation_ids") == INSTANCE_IDS, "Controlled instance ID getter audit differs")
    _require(report.get("segmentation_palette_sha256") == _sha(palette_path)
             and report.get("segmentation_palette_rgb") == palette, "Independent segmentation palette hash/colors differ")
    _ground(report.get("ground_before"), "Ground before")
    _ground(report.get("ground_after"), "Ground after")
    expected_sources = row.get("expected_source_sha256", plan.get("expected_source_sha256"))
    _require(isinstance(expected_sources, dict) and REQUIRED_SOURCES <= set(expected_sources),
             "Frozen probe/setup_scene/camera_geometry source hashes required")
    recorded_sources = report.get("source_sha256")
    _require(isinstance(recorded_sources, dict) and REQUIRED_SOURCES <= set(recorded_sources),
             "Report source hash inventory missing")
    snapshot = _path(row["source_snapshot_dir"], base, "source_snapshot_dir") if row.get("source_snapshot_dir") else manifest_path.parent / "source_snapshot"
    for name, expected in expected_sources.items():
        _identifier(name, "source filename")
        _hash(expected, "frozen source SHA256")
        _require(recorded_sources.get(name) == expected, "Captured source differs from preregistration: " + name)
    for name, expected in recorded_sources.items():
        _identifier(name, "source filename")
        _require(_sha(snapshot / name) == _hash(expected, "report source SHA256"), "Source snapshot hash mismatch: " + name)
    launcher_hash = _hash(manifest.get("launcher_sha256"), "launcher SHA256")
    launchers = [path for path in snapshot.glob("*.ps1") if _sha(path) == launcher_hash]
    _require(len(launchers) == 1, "Exactly one hash-matching archived launcher required")
    if "launcher_sha256" in plan:
        _require(launcher_hash == _hash(plan["launcher_sha256"], "preregistered launcher SHA256"),
                 "Launcher differs from preregistration")
    if "plan_sha256" in manifest or "plan" in manifest:
        _require(manifest.get("plan_sha256") == _sha(plan_path)
                 and _path(manifest.get("plan"), manifest_path.parent, "launcher plan") == plan_path,
                 "Launcher did not use the current frozen plan")
    settings_path = _path(manifest.get("settings"), manifest_path.parent, "launcher settings")
    settings_hash = _hash(manifest.get("settings_sha256"), "launcher settings SHA256")
    _require(_sha(settings_path) == settings_hash, "Settings hash mismatch")
    if "settings_path" in plan or "settings_sha256" in plan:
        _require(settings_path == _path(plan.get("settings_path"), base, "preregistered settings")
                 and settings_hash == plan.get("settings_sha256"), "Settings differ from preregistration")
    cases = protocol.get("cases")
    _require(isinstance(cases, list) and bool(cases), "Nonempty registered cases required")
    case_map = {}
    for case in cases:
        _require(isinstance(case, dict), "Invalid protocol case")
        name = _identifier(case.get("case_id"), "case_id")
        _require(name not in case_map, "Duplicate registered case_id")
        case_map[name] = case
    repeats = _positive_int(protocol.get("frames_per_case"), "frames_per_case")
    frames = report.get("frames")
    _require(isinstance(frames, list) and len(frames) == len(cases) * repeats,
             "Complete registered cases × frames are required")
    _require(manifest.get("saved_frames") == len(frames), "Launcher frame count differs from report")
    if "cases_per_episode" in plan:
        _require(plan["cases_per_episode"] == len(cases), "Case count differs from preregistration")
    if "frames_per_case" in plan:
        _require(plan["frames_per_case"] == repeats, "Frame repetitions differ from preregistration")
    expected_ids = {f"{case_id}_{repeat:02d}" for case_id in case_map for repeat in range(1, repeats + 1)}
    seen_ids, sessions, prepared = set(), set(), []
    for frame in frames:
        frame_id = _identifier(frame.get("frame_id"), "frame_id")
        _require(frame_id in expected_ids and frame_id not in seen_ids, "Unexpected/duplicate registered frame_id")
        seen_ids.add(frame_id)
        case_id = frame.get("case_id")
        _require(case_id in case_map and any(frame_id == f"{case_id}_{repeat:02d}" for repeat in range(1, repeats + 1)),
                 "Frame case/id association differs")
        case = case_map[case_id]
        session = frame.get("session_id")
        _require(isinstance(session, str) and bool(session) and session not in excluded["sessions"],
                 "Frame session absent or already consumed in development")
        sessions.add(session)
        _require(frame.get("scene_acceptance") == "PASS_INTENDED_VISIBLE_INSTANCES",
                 "Frame lacks collector scene acceptance")
        _ground(frame.get("ground_state"), "Frame ground state")
        size = frame.get("image_size_wh")
        _require(isinstance(size, list) and len(size) == 2, "Image dimensions missing")
        width, height = [_positive_int(value, "image dimension") for value in size]
        media = frame.get("media")
        _require(isinstance(media, dict) and set(media) == {"rgb", "depth", "segmentation"},
                 "Paired RGB/depth/segmentation media required")
        paths = {}
        for kind, receipt in media.items():
            path = _path(receipt.get("path"), report_path.parent, "media path")
            _require(_sha(path) == _hash(receipt.get("sha256"), "media SHA256"), "Raw media hash mismatch: " + kind)
            paths[kind] = path
        rgb, segmentation = _image(paths["rgb"]), _image(paths["segmentation"])
        _require(rgb.shape == segmentation.shape and rgb.shape[:2] == (height, width),
                 "Label image dimensions differ from RGB")
        depth = np.load(paths["depth"], allow_pickle=False)
        _require(depth.shape == (height, width) and np.issubdtype(depth.dtype, np.number),
                 "Paired depth dimensions/type differ")
        observed = {name: visible_mask(segmentation, palette[str(segid)])
                    for name, segid in (("target", 42), ("distractor", 43), ("occluder", 44))}
        truth = frame.get("evaluation_only")
        _require(isinstance(truth, dict) and all(observed[name] == truth.get(name) for name in observed),
                 "Stored instance masks do not independently reproduce")
        _scene(case, observed)
        labels, instances = [], []
        for name, segid in (("target", 42), ("distractor", 43)):
            if observed[name]["bbox_xyxy_pixels"] is not None:
                labels.append(yolo_label(observed[name]["bbox_xyxy_pixels"], width, height))
                instances.append({"evaluation_identity": name, "segmentation_id": segid, **observed[name]})
        label_bytes = _label_text(labels).encode("utf-8")
        raw_hash, pixel_hash = media["rgb"]["sha256"], _pixel_sha(rgb)
        _require(raw_hash not in excluded["rgb_hashes"] and pixel_hash not in excluded["pixel_hashes"],
                 "RGB was already consumed by the development set")
        prepared.append({"episode_id": episode_id, "source_session_id": session, "split": split,
            "yolo_directory_split": YOLO_SPLIT[split], "acquisition_seed": seed,
            "case_id": case_id, "frame_id": frame_id, "image_size_wh": size,
            "source_rgb_path": str(paths["rgb"]), "source_rgb_sha256": raw_hash,
            "rgb_sha256": raw_hash, "decoded_rgb_sha256": pixel_hash,
            "source_segmentation_path": str(paths["segmentation"]),
            "source_segmentation_sha256": media["segmentation"]["sha256"],
            "source_depth_path": str(paths["depth"]), "source_depth_sha256": media["depth"]["sha256"],
            "source_label_kind": "Independent visible-mask half-open boxes; no original label file",
            "source_label_sha256": _bytes_sha(label_bytes), "visible_balloon_count": len(labels),
            "instances_training_supervision_only": instances, "scene_acceptance_recomputed": "PASS",
            "_label_bytes": label_bytes})
    _require(seen_ids == expected_ids and len(sessions) == 1,
             "Exactly one complete rendered session per registered episode is required")
    return {"episode_id": episode_id, "split": split, "acquisition_seed": seed,
            "UE_process_id": process[0], "UE_started_utc": manifest["engine_started_utc"],
            "UE_process_identity": list(process), "source_session_id": next(iter(sessions)),
            "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path),
            "report_path": str(report_path), "report_sha256": _sha(report_path),
            "protocol_path": str(protocol_path), "protocol_sha256": protocol_hash,
            "source_snapshot_dir": str(snapshot), "source_sha256": recorded_sources,
            "launcher_snapshot": str(launchers[0]), "launcher_sha256": launcher_hash,
            "frame_count": len(prepared), "registered_case_count": len(cases),
            "frames_per_case": repeats, "safe_ground_closure": True}, prepared


def _validate(plan_path, output):
    plan = _load(plan_path)
    _require(plan.get("status") == "FROZEN_BEFORE_COLLECTION_AND_TRAINING" and plan.get("ground_only") is True,
             "Frozen-before-collection ground-only plan required")
    _require(not output.exists(), "Dataset destination already exists; preserve it")
    base = plan_path.parent
    episodes = plan.get("episodes", plan.get("rows"))
    _require(isinstance(episodes, list) and bool(episodes), "Registered episode rows required")
    palette_path = _path(plan.get("segmentation_palette_path", str(ROOT / "external/AirSim/docs/seg_rgbs.txt")),
                         base, "segmentation_palette_path")
    palette = _palette(palette_path)
    excluded = _exclusions(plan, base)
    provenance, prepared = [], []
    episode_ids, processes, sessions, report_paths, protocol_ids, seeds = set(), set(), set(), set(), set(), set()
    for row in episodes:
        _require(isinstance(row, dict), "Invalid registered episode row")
        episode, frames = _validate_episode(row, plan, base, palette_path, palette, excluded, plan_path)
        _require(episode["episode_id"] not in episode_ids, "An episode is repeated in the campaign plan")
        _require(tuple(episode["UE_process_identity"]) not in processes, "A UE process is shared by multiple episodes/splits")
        _require(episode["source_session_id"] not in sessions, "A rendered session is shared by multiple episodes/splits")
        _require(episode["report_path"] not in report_paths, "A report is shared by multiple episodes")
        _require(episode["acquisition_seed"] not in seeds, "Acquisition seeds must be distinct across independent episodes")
        protocol_id = _load(episode["protocol_path"]).get("protocol_id")
        _require(isinstance(protocol_id, str) and protocol_id and protocol_id not in protocol_ids,
                 "Registered protocol_id must be distinct across episodes")
        episode_ids.add(episode["episode_id"]); processes.add(tuple(episode["UE_process_identity"]))
        sessions.add(episode["source_session_id"]); report_paths.add(episode["report_path"])
        seeds.add(episode["acquisition_seed"]); protocol_ids.add(protocol_id)
        provenance.append(episode); prepared.extend(frames)
    episode_counts = validate_episode_split(prepared)
    validate_episode_split([dict(row, rgb_sha256=row["decoded_rgb_sha256"]) for row in prepared])
    first_images, first_pixels = {}, {}
    for row in prepared:
        for identity, observed in ((row["rgb_sha256"], first_images), (row["decoded_rgb_sha256"], first_pixels)):
            if identity in observed:
                first = observed[identity]
                _require(first["split"] == row["split"], "Identical RGB crosses dataset splits")
                _require(first["source_label_sha256"] == row["source_label_sha256"],
                         "Identical RGB has contradictory labels")
            else:
                observed[identity] = row
        original = first_pixels[row["decoded_rgb_sha256"]]
        row["duplicate_rgb_within_split"] = original is not row
        row["duplicate_of"] = (original["episode_id"] + "__" + original["frame_id"]) if original is not row else None
    return plan, provenance, prepared, episode_counts, excluded, palette_path


def _failure_receipt(output, phase, exc, plan_path, checked_episodes=0, created_output=False):
    """Outside the proposed dataset; never replace an earlier failure or dataset."""
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    receipt_path = parent / (output.name + ".validation_failure_" + uuid.uuid4().hex + ".json")
    value = {"status": "FAIL", "phase": phase, "dataset_output_created": created_output,
             "dataset_path_exists": output.exists(),
             "dataset_path": str(output), "plan_path": str(plan_path), "error_type": type(exc).__name__,
             "reason": str(exc), "checked_episode_count": checked_episodes,
             "recorded_utc": datetime.now(timezone.utc).isoformat(),
             "no_training_or_model_calls": True,
             "ai_assistance": "AI assisted validation and controlled-label dataset implementation"}
    if plan_path.is_file():
        value["plan_sha256"] = _sha(plan_path)
    _save(receipt_path, value)
    return receipt_path


def build(campaign_plan_path, output):
    """Validate all frozen episodes and export a new dataset, with no inference.

    Plan rows require episode_id, split, acquisition_seed, protocol_path,
    protocol_sha256, manifest_path and report_path. Paths are absolute or relative
    to the plan. Top-level expected_source_sha256 freezes collector dependencies.
    Validation receipts are kept next to output when preflight rejects a build.
    """
    plan_path, output = Path(campaign_plan_path).resolve(), Path(output).resolve()
    try:
        plan, episodes, prepared, episode_counts, excluded, palette_path = _validate(plan_path, output)
    except Exception as exc:
        receipt = _failure_receipt(output, "PRE_EXPORT_VALIDATION", exc, plan_path)
        raise DatasetBuildError(str(exc), receipt) from exc
    manifest = {"status": "EXPORTING", "kind": "T1_CONTROLLED_EPISODE_GROUPED_BALLOON_DATASET",
        "plan_path": str(plan_path), "plan_sha256": _sha(plan_path),
        "source_sha256": {Path(__file__).name: _sha(__file__)},
        "class_names": {"0": "balloon"}, "target_and_distractor_share_class": True,
        "split_unit": "Whole independently launched UE process/session; preregistered acquisition seeds",
        "split_directory_mapping": YOLO_SPLIT, "independent_UE_sessions": len(episodes),
        "train_val_test_ready": True, "formal_efficacy_results": False,
        "segmentation_palette_path": str(palette_path), "segmentation_palette_sha256": _sha(palette_path),
        "episodes": episodes, "episode_counts_by_split": episode_counts,
        "excluded_development_provenance": excluded["receipts"], "rows": [],
        "model_calls": 0, "training_executed": False,
        "label_truth_access": "Training supervision/evaluation only; exported model input is RGB only",
        "limits": ["Controlled same-map/sphere-mesh/red-material adaptation pilot",
            "Whole UE sessions and distinct acquisition seeds separate splits; sessions share scene assets",
            "Visible-mask boxes are not amodal human labels and do not establish instance tracking",
            "No map/material/real-world generalization or formal thesis trajectory effectiveness claimed",
            "No training, model inference, cloud request or new flight is performed by this builder"],
        "ai_assistance": "AI assisted builder, provenance checks and label export"}
    created_output = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        for split in YOLO_SPLIT.values():
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
        _save(output / "manifest.json", manifest)
        for row in prepared:
            # Recheck originals immediately before copying to catch post-validation changes.
            _require(_sha(row["source_rgb_path"]) == row["source_rgb_sha256"], "RGB changed after preflight")
            name = row["episode_id"] + "__" + row["frame_id"]
            image_path = output / "images" / row["yolo_directory_split"] / (name + ".png")
            label_path = output / "labels" / row["yolo_directory_split"] / (name + ".txt")
            shutil.copyfile(row["source_rgb_path"], image_path)
            label_path.write_bytes(row["_label_bytes"])
            _require(_sha(image_path) == row["source_rgb_sha256"]
                     and _sha(label_path) == row["source_label_sha256"], "Export hash verification failed")
            manifest["rows"].append({key: value for key, value in row.items() if not key.startswith("_")} | {
                "image": str(image_path), "image_relative": image_path.relative_to(output).as_posix(),
                "image_path": str(image_path), "image_sha256": _sha(image_path),
                "exported_rgb_sha256": _sha(image_path), "label": str(label_path), "label_path": str(label_path),
                "label_relative": label_path.relative_to(output).as_posix(),
                "label_sha256": _sha(label_path), "exported_label_sha256": _sha(label_path),
                "visible_boxes_xyxy_pixels": [instance["bbox_xyxy_pixels"] for instance in row["instances_training_supervision_only"]]})
        counts = {}
        for split in SPLITS:
            rows = [row for row in manifest["rows"] if row["split"] == split]
            counts[split] = {"episode_count": episode_counts[split], "frame_count": len(rows),
                "visible_instance_labels": sum(row["visible_balloon_count"] for row in rows),
                "negative_frame_count": sum(row["visible_balloon_count"] == 0 for row in rows),
                "unique_rgb_file_hashes": len({row["rgb_sha256"] for row in rows}),
                "unique_decoded_rgb_hashes": len({row["decoded_rgb_sha256"] for row in rows}),
                "duplicate_rgb_frame_count": sum(row["duplicate_rgb_within_split"] for row in rows)}
        yaml_path = output / "dataset.yaml"
        yaml_path.write_text("# RGB input only; class-0 visible-mask balloon supervision.\n"
            + "path: " + json.dumps(output.as_posix(), ensure_ascii=False) + "\n"
            + "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: balloon\n",
            encoding="utf-8")
        manifest.update(status="PASS", frame_count=len(manifest["rows"]), splits=counts,
            visible_instance_labels=sum(row["visible_balloon_count"] for row in manifest["rows"]),
            negative_frame_count=sum(row["visible_balloon_count"] == 0 for row in manifest["rows"]),
            duplicate_rgb_frame_count=sum(row["duplicate_rgb_within_split"] for row in manifest["rows"]),
            dataset_yaml=str(yaml_path), dataset_yaml_sha256=_sha(yaml_path),
            completed_utc=datetime.now(timezone.utc).isoformat())
        manifest["frames"] = manifest["rows"]  # Explicit compatibility alias for training/evaluation consumers.
        _save(output / "manifest.json", manifest)
        (output / "README.md").write_text(
            "# 按独立会话分组的气球适配数据\n\n"
            "train、validation（YOLO 路径 val）、test 由采集前冻结的 episode/种子指定。"
            "每个 episode 来自独立启动并安全关闭的 UE 进程；相邻帧不随机拆分。"
            "同组重复图像保留并计数，跨组相同原图或解码 RGB 被拒绝。\n\n"
            "类 0 balloon 包含指定目标与干扰气球；标签用实例分割调色板独立复算的可见半开框，"
            "全遮挡且无可见气球的负例为空标签。images 仅含 RGB，掩膜不输入模型。\n\n"
            "这是同地图、同球体网格、同红色材质的小规模适配集，不代表泛化、身份跟踪或论文轨迹效果。"
            "本构建步骤没有训练、推理、API 调用或飞行。\n\n"
            "AI 使用说明：代码与证据整理由 AI 辅助编制。原始/导出图像与标签摘要、协议、进程来源、"
            "场景准入和分组计数见 manifest.json。\n", encoding="utf-8")
    except Exception as exc:
        manifest.update(status="FAIL", error_type=type(exc).__name__, reason=str(exc))
        if created_output:
            _save(output / "manifest.json", manifest)
        receipt = _failure_receipt(output, "EXPORT", exc, plan_path, len(episodes), created_output)
        raise DatasetBuildError(str(exc), receipt) from exc
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.plan, args.output)
    except DatasetBuildError as error:
        print(json.dumps({"status": "FAIL", "reason": str(error), "receipt": str(error.receipt_path)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps({"status": result["status"], "frames": result["frame_count"],
                      "episodes": result["independent_UE_sessions"], "manifest": str(args.output / "manifest.json")}, ensure_ascii=False))

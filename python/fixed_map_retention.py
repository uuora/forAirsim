"""Remove unselected raw captures only after verified postflight shutdown.

Command/plan evidence and selected frames remain replayable. Purged frames keep
their original metadata and hashes, but their full raw replay is unavailable.
"""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import os
import re
import stat


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RAW_BASENAMES = frozenset(('rgb.png', 'depth_ray_range_m.npy', 'overlay.jpg', 'overlay.png'))
EXCLUDED_TELEMETRY_KINDS = frozenset(('PERCEPTION_CONTROL_CYCLE_TIMING',))
FRAME_NAME = re.compile(r'^frame-[0-9]+$')
FRAME_REFERENCE = re.compile(r'(?<![A-Za-z0-9_-])(frame-[0-9]+)(?![A-Za-z0-9_-])')
EXPLICIT_SELECTORS = frozenset(('result.json', 'final_ground_receipt.json', 'live_status.json', 'trial_status.json', 'language.json'))
SELECTION_WORDS = frozenset(('selected', 'positive', 'reference', 'accepted', 'click', 'source'))


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _no_reparse(path):
    info = path.lstat()
    if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
        raise ValueError('RAW_RETENTION_REPARSE_PATH_REFUSED: ' + str(path))
    return info


def _safe_path(path, root, require_file=False):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError('RAW_RETENTION_PATH_OUTSIDE_OUTPUT')
    # Check each literal component before resolving it; Windows junctions
    # are reparse points even when Path.is_symlink() is false.
    for component in (root, *reversed(path.relative_to(root).parents[:-1])):
        check = component if component.is_absolute() else root / component
        if os.path.lexists(check):
            _no_reparse(check)
    if os.path.lexists(path):
        info = _no_reparse(path)
        if require_file and not stat.S_ISREG(info.st_mode):
            raise ValueError('RAW_RETENTION_REQUIRES_REGULAR_FILE: ' + str(path))
    if not path.resolve(strict=False).is_relative_to(root):
        raise ValueError('RAW_RETENTION_RESOLVED_PATH_OUTSIDE_OUTPUT')
    return path


def _output_root(output):
    literal = Path(os.path.abspath(output))
    workspace = WORKSPACE_ROOT.resolve()
    if workspace.drive.upper() != 'D:' or not literal.is_relative_to(workspace):
        raise ValueError('RAW_RETENTION_OUTPUT_MUST_STAY_IN_D_WORKSPACE')
    for component in reversed((literal, *literal.parents)):
        if os.path.lexists(component):
            _no_reparse(component)
    root = literal.resolve(strict=True)
    if not root.is_relative_to(workspace):
        raise ValueError('RAW_RETENTION_OUTPUT_OUTSIDE_WORKSPACE')
    return root


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(root, name, value):
    path = _safe_path(root / name, root)
    temporary = _safe_path(root / (name + '.tmp'), root)
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    _safe_path(path, root)
    temporary.replace(path)


def _referenced_frames(value):
    found = set()
    if isinstance(value, dict):
        for nested in value.values():
            found.update(_referenced_frames(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_referenced_frames(nested))
    elif isinstance(value, str):
        found.update(FRAME_REFERENCE.findall(unquote(value)))
    return found


def _selector_json(name):
    name = name.lower()
    if not name.endswith('.json'):
        return False
    if name in EXPLICIT_SELECTORS or name.startswith('perception_plan_'):
        return True
    words = set(re.split(r'[_\-.]+', name))
    # Publication/capture indexes and diagnostic metric collections are not
    # raw-evidence selectors, even though their metadata files are retained.
    if words & {'index', 'indexes', 'trace', 'traces', 'metrics', 'publication'}:
        return False
    return bool(words & SELECTION_WORDS)


def _selector_files(root, frames_root):
    paths = [root / 'session_events.jsonl']
    if not paths[0].is_file():
        raise ValueError('RAW_RETENTION_SESSION_EVENT_SELECTOR_MISSING')
    def walk_error(error):
        raise error
    for current, directories, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        current = Path(current)
        _safe_path(current, root)
        kept = []
        for name in directories:
            child = current / name
            _safe_path(child, root)
            if child != frames_root:
                kept.append(name)
        directories[:] = kept
        for name in filenames:
            if _selector_json(name):
                paths.append(_safe_path(current / name, root, require_file=True))
    return sorted(set(paths), key=lambda path: str(path.relative_to(root)))


def _ground_shutdown_gate(terminal, owner_stage, owner_alive):
    vehicles = terminal.get('final_ground', {})
    no_flight_worker_started = (
        terminal.get('flight_start_requested') is False
        and owner_stage == 'CHECKING_NO_FLIGHT_NO_OWNER_THREAD'
        and owner_alive is False
        and not (Path(terminal.get('output_path', '')) / 'online' / 'frames').exists()
    )
    if (terminal.get('ground_verification_passed') is not True
            or set(vehicles) != {'DroneA', 'DroneB'}
            or not all(v.get('landed') is True and v.get('api_control') is False and v.get('fresh') is True for v in vehicles.values())
            or (terminal.get('sensor_cleanup', {}).get('status') != 'PASS' and not no_flight_worker_started)
            or terminal.get('terminal_phase') != 'FINAL'
            or (owner_stage != 'READY' and not no_flight_worker_started) or owner_alive is not False
            or terminal.get('mission', {}).get('active') is True):
        raise ValueError('RAW_RETENTION_REQUIRES_VERIFIED_GROUND_AND_SENSOR_OWNER_SHUTDOWN')


def apply_postflight_raw_retention(output, terminal, *, owner_stage, owner_alive):
    """Hash a complete deletion plan before unlinking strict raw filenames."""
    _ground_shutdown_gate(terminal, owner_stage, owner_alive)
    root = _output_root(output)
    frames_root = _safe_path(root / 'online' / 'frames', root)
    receipt = dict(
        schema='postflight_raw_retention_v1', utc=_utc(), status='PASS',
        output_path=str(root), verified_ground=True,
        sensor_cleanup_status=(terminal.get('sensor_cleanup', {}).get('status')
                               or 'NOT_STARTED_NO_FLIGHT_NO_SENSOR_OWNER'),
        owner_stage=owner_stage, owner_alive=False,
        resource_trace_status=terminal.get('resource_trace', {}).get('status', 'NOT_CONFIGURED'),
        raw_basenames_allowlist=sorted(RAW_BASENAMES),
        excluded_event_telemetry_kinds=sorted(EXCLUDED_TELEMETRY_KINDS),
        all_metadata_and_highlevel_json_retained=True,
        selector_policy='all non-telemetry session event references; terminal/status/plan/explicit selection JSON; every perception.json with valid exactly true; first and last two frame folders',
        raw_replay_scope='retained frames only; purged frames cannot support full raw video or full-frame accuracy reruns',
        automatic_training_code_model_deletion=False,
        raw_payload_purged=False, purged_file_count=0, purged_bytes=0,
        purged_frame_ids=[], errors=[],
    )
    if not frames_root.exists():
        receipt.update(retained_frame_ids=[], selector_files=[], frame_folder_count=0)
        _json_write(root, 'RAW_RETENTION_RECEIPT.json', receipt)
        return receipt
    frame_directories = {}
    for child in frames_root.iterdir():
        if FRAME_NAME.fullmatch(child.name):
            _safe_path(child, root)
            if not child.is_dir():
                raise ValueError('RAW_RETENTION_FRAME_FOLDER_NOT_DIRECTORY')
            frame_directories[child.name] = child
    referenced = set()
    selector_receipts = []
    skipped_events = 0
    for path in _selector_files(root, frames_root):
        _safe_path(path, root, require_file=True)
        selected = set()
        if path.name == 'session_events.jsonl' and path.parent == root:
            with path.open('r', encoding='utf-8-sig') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get('kind') in EXCLUDED_TELEMETRY_KINDS:
                        skipped_events += 1
                        continue
                    selected.update(_referenced_frames(event))
        else:
            selected = _referenced_frames(json.loads(path.read_text(encoding='utf-8-sig')))
        referenced.update(selected)
        selector_receipts.append(dict(path=str(path.relative_to(root)), sha256=_sha(path), referenced_frame_ids=sorted(selected)))
    # The postflight auditor revalidates the whole originally-valid population,
    # not only command frames. Keep its original RGB/depth certificates intact.
    certificate_frames = set()
    for frame_id, folder in frame_directories.items():
        certificate = folder / 'perception.json'
        if not os.path.lexists(certificate):
            continue
        _safe_path(certificate, root, require_file=True)
        report = json.loads(certificate.read_text(encoding='utf-8-sig'))
        if report.get('valid') is True:
            certificate_frames.add(frame_id)
            selected = _referenced_frames(report) | {frame_id}
            referenced.update(selected)
            selector_receipts.append(dict(
                path=str(certificate.relative_to(root)), sha256=_sha(certificate),
                referenced_frame_ids=sorted(selected), original_valid=True,
                selection_reason='originally_valid_perception_certificate_for_postflight_audit',
            ))
    ordered = sorted(frame_directories, key=lambda name: int(name.split('-')[1]))
    boundary = set(ordered[:2] + ordered[-2:])
    retained = referenced | boundary
    receipt.update(
        frame_folder_count=len(ordered), selector_files=selector_receipts,
        selector_observation='files read after final result persistence, before raw purge',
        excluded_telemetry_event_count=skipped_events,
        originally_valid_certificate_frame_ids=sorted(certificate_frames),
        originally_valid_certificate_frame_count=len(certificate_frames),
        originally_valid_certificate_retention_reason='preserve original RGB/depth for audit of every recorded_valid exactly true perception.json',
        retained_frame_ids=sorted(retained & set(ordered)),
        referenced_frame_ids=sorted(referenced), boundary_frame_ids=sorted(boundary),
        referenced_frame_ids_not_in_this_capture_tree=sorted(referenced - set(ordered)),
    )
    manifest = _safe_path(root / 'RAW_RETENTION_MANIFEST.jsonl', root)
    if manifest.exists():
        raise ValueError('RAW_RETENTION_MANIFEST_ALREADY_EXISTS_NO_SECOND_PURGE')
    planned = []
    for frame_id in ordered:
        if frame_id in retained:
            continue
        for basename in sorted(RAW_BASENAMES):
            path = frame_directories[frame_id] / basename
            if not path.exists():
                continue
            _safe_path(path, root, require_file=True)
            info = path.lstat()
            record = dict(kind='RAW_PAYLOAD_DELETE_PLANNED', utc=_utc(), frame_id=frame_id,
                          path=str(path.relative_to(root)), bytes=info.st_size, sha256=_sha(path),
                          mtime_ns=info.st_mtime_ns, inode=info.st_ino, device=info.st_dev,
                          metadata_retained=True, raw_purged=False)
            planned.append(record)
    # Every candidate's bytes/hash/path is durable before the first unlink.
    with manifest.open('x', encoding='utf-8') as handle:
        for record in planned:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    purged_frames = set()
    _safe_path(manifest, root, require_file=True)
    with manifest.open('a', encoding='utf-8') as handle:
        for record in planned:
            try:
                path = _safe_path(root / record['path'], root, require_file=True)
                if path.name not in RAW_BASENAMES or path.parent.parent != frames_root:
                    raise ValueError('RAW_RETENTION_STRICT_RAW_PATH_GATE_FAILED')
                info = path.lstat()
                if (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev) != (record['bytes'], record['mtime_ns'], record['inode'], record['device']):
                    raise ValueError('RAW_RETENTION_PAYLOAD_CHANGED_AFTER_MANIFEST')
                path.unlink()
                receipt['purged_file_count'] += 1
                receipt['purged_bytes'] += record['bytes']
                purged_frames.add(record['frame_id'])
                handle.write(json.dumps(dict(record, kind='RAW_PAYLOAD_PURGED', utc=_utc(), raw_purged=True), ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
            except Exception as ex:
                receipt['status'] = 'FAILED'
                receipt['errors'].append(dict(path=record['path'], error_type=type(ex).__name__, error=str(ex)))
                break
    receipt.update(utc=_utc(), raw_payload_purged=bool(receipt['purged_file_count']),
                   purged_frame_ids=sorted(purged_frames),
                   manifest_path=str(manifest.relative_to(root)),
                   manifest_sha256=_sha(manifest), planned_purge_file_count=len(planned),
                   planned_purge_bytes=sum(record['bytes'] for record in planned))
    _json_write(root, 'RAW_RETENTION_RECEIPT.json', receipt)
    return receipt

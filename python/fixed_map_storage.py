"""Fixed output-volume space reserves for the persistent experiment owner.

This guard never deletes evidence, changes capture frequency, or terminates a
flight owner. Its measurements concern the volume holding the actual output
directory; storage on other volumes is outside this guard's scope.
"""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time


PREFLIGHT_REQUIRED_BYTES = 4 * 2**30
ACTIVE_RESERVE_BYTES = 2 * 2**30
CHECK_PERIOD_HOST_S = .5


def storage_policy():
    """Thresholds are fixed in source, with no configuration override."""
    return dict(
        schema='output_volume_reserve_v1',
        unconditional=True,
        preflight_required_free_bytes=PREFLIGHT_REQUIRED_BYTES,
        active_required_free_bytes=ACTIVE_RESERVE_BYTES,
        active_check_period_host_s=CHECK_PERIOD_HOST_S,
        low_space_action='cancel_sensor_mission_stop_and_return_on_existing_owner',
        unavailable_measurement_action='reject_ground_start_or_request_safe_return',
        output_volume_only=True,
        automatic_evidence_deletion=False,
        capture_policy_changed=False,
        configurable_threshold_override=False,
    )


def check_output_storage(output, phase):
    """Return an honest verdict; an unavailable probe is never free space."""
    if phase not in ('preflight', 'active'):
        raise ValueError('STORAGE_CHECK_PHASE_INVALID')
    path = Path(output).resolve()
    required = PREFLIGHT_REQUIRED_BYTES if phase == 'preflight' else ACTIVE_RESERVE_BYTES
    receipt = dict(
        schema='output_volume_storage_check_v1',
        utc=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        checked_monotonic_s=time.monotonic(),
        phase=phase,
        output_path=str(path),
        queried_path=str(path),
        required_free_bytes=required,
        policy=storage_policy(),
    )
    try:
        # Query this resolved directory, rather than assuming its drive letter
        # identifies the filesystem (junctions/mounted volumes may differ).
        usage = shutil.disk_usage(path)
    except OSError as ex:
        receipt.update(status='UNAVAILABLE', error_type=type(ex).__name__, error=str(ex))
    else:
        receipt.update(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            status='PASS' if usage.free >= required else 'LOW_SPACE',
        )
    return receipt

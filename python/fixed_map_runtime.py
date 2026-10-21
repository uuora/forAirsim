"""Relocate mutable mission records while retaining the original frozen inputs."""
from pathlib import Path
import shutil
import fixed_map_mission as mission
def configure_session(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    historical=mission.ROOT/'reports/fixed_map_flight_20261009'
    if output==historical or output.is_relative_to(historical):raise ValueError('HISTORICAL_OUTPUT_IS_READ_ONLY')
    # The executor expects OUT/fixed_version. Copy only small immutable inputs;
    # inventory records continue to reference and verify the old large snapshots.
    for rel in ['manifest.json','task.json','editor_audit/planning_geometry.json','editor_audit/scene_geometry.json','snapshot/configs/enclosed_uav_lab_manual_settings.json']:
        source=historical/'fixed_version'/rel;dest=output/'fixed_version'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(source,dest)
        elif mission.sha(dest)!=mission.sha(source):raise ValueError('SESSION_FROZEN_INPUT_MISMATCH')
    mission.OUT=output
    return output

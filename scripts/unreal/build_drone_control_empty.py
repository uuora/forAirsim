"""Build only /Game/DroneControl: one collision floor and one player start."""
import json
import traceback
from pathlib import Path
import unreal as ue

ROOT = Path(__file__).resolve().parents[2]
MAP = '/Game/DroneControl/Maps/DroneControlEmpty'
receipt = {'map': MAP, 'flight_commands': False, 'status': 'BUILDING'}
try:
    # An existing isolated map is reusable; no other level is loaded or edited.
    if ue.EditorAssetLibrary.does_asset_exist(MAP):
        if not ue.EditorLevelLibrary.load_level(MAP):
            raise RuntimeError('Cannot load DroneControlEmpty')
    else:
        if not ue.EditorLevelLibrary.new_level(MAP):
            raise RuntimeError('Cannot create DroneControlEmpty')
        world = ue.EditorLevelLibrary.get_editor_world()
        mode = ue.load_class(None, '/Script/AirSim.AirSimGameMode')
        if not mode:
            raise RuntimeError('AirSim game mode unavailable')
        world.get_world_settings().set_editor_property('default_game_mode', mode)
        start = ue.EditorLevelLibrary.spawn_actor_from_class(ue.PlayerStart, ue.Vector(0, 0, 100))
        start.set_actor_label('DC_PlayerStart')
        floor = ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor, ue.Vector(0, 0, -50))
        floor.set_actor_label('DC_CollisionFloor')
        floor.static_mesh_component.set_static_mesh(ue.load_asset('/Engine/BasicShapes/Cube'))
        floor.set_actor_scale3d(ue.Vector(1000, 1000, 1))
        floor.set_actor_hidden_in_game(True)
        floor.static_mesh_component.set_collision_profile_name('BlockAll')
        floor.static_mesh_component.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS)
        floor.static_mesh_component.set_editor_property('cast_shadow', False)
        # No sky, buildings, lights, models or visible target actors are necessary.
        if not ue.EditorLevelLibrary.save_current_level():
            raise RuntimeError('Cannot save DroneControlEmpty')
    actors = ue.EditorLevelLibrary.get_all_level_actors()
    labels = [a.get_actor_label() for a in actors]
    if 'DC_CollisionFloor' not in labels or 'DC_PlayerStart' not in labels:
        raise RuntimeError('Required floor/start absent')
    receipt.update(status='PASS', actor_labels=labels)
except Exception:
    receipt.update(status='FAIL', error=traceback.format_exc())
finally:
    out = ROOT / 'logs/dual_target_control_build.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    ue.log(str(receipt))

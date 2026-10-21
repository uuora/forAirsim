"""Read the configured AirSim pawn bounds without saving or modifying the map."""
import unreal as ue
import json
from pathlib import Path

out=Path('D:/forAirsim/reports/fixed_map_flight_20261009')
ue.EditorLevelLibrary.load_level('/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual')
bp=ue.load_object(None,'/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn')
pawn=ue.EditorLevelLibrary.spawn_actor_from_class(ue.load_class(None,'/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn_C'),ue.Vector(0,0,1000))
def vec(v): return [float(v.x)/100,float(v.y)/100,float(v.z)/100]
origin,extent=pawn.get_actor_bounds(False)
rows=[]
for c in pawn.get_components_by_class(ue.StaticMeshComponent):
    mesh=c.get_editor_property('static_mesh')
    rows.append(dict(name=c.get_name(),mesh=mesh.get_path_name() if mesh else None,
                     collision=str(c.get_collision_enabled()),
                     relative_location_m=vec(c.get_editor_property('relative_location'))))
result=dict(actor_class=bp.get_path_name(),visual_bounds_origin_world_m=vec(origin),
            visual_bounds_half_extent_m=vec(extent),pawn_origin_world_m=vec(pawn.get_actor_location()),components=rows)
(out/'drone_envelope_editor.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
ue.EditorLevelLibrary.destroy_actor(pawn)
ue.log('ENVELOPE_READ_ONLY_COMPLETE')

"""Install an off-field research sphere into the exact 03 map, with backup.

Original dual-target functionality is unchanged. Only research runs relocate
this reserved actor; it is stored far below/outside the normal flight field.
"""
import hashlib,json,shutil,traceback
from pathlib import Path
import unreal as ue
ROOT=Path('D:/forAirsim');MAP='/Game/DroneControlOverview/Maps/DroneControlOverview'
source=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/DroneControlOverview/Maps/DroneControlOverview.umap'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'reports/curve_sphere_install_overview_v1.json'
report=dict(status='RUNNING',map=MAP,flight_commands=0)
try:
    prior=json.loads(out.read_text()) if out.is_file() else None
    if prior and prior.get('status')=='PASS':raise RuntimeError('already installed; use existing receipt')
    backup=ROOT/'reports/curve_sphere_overview_original_v1.umap'
    if not backup.exists():shutil.copy2(source,backup)
    report.update(original_map_backup=str(backup),original_sha256=sha(backup))
    if not ue.EditorLevelLibrary.load_level(MAP):raise RuntimeError('03 map not loaded')
    if any(a.get_actor_label()=='CurveSphere_Central' for a in ue.EditorLevelLibrary.get_all_level_actors()):
        raise RuntimeError('refuse duplicate/unowned research sphere')
    mesh=ue.load_asset('/Game/CurveControl/Meshes/SM_CurveControlSphere_v1')
    actor=ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor,ue.Vector(100000,100000,-50000))
    actor.set_actor_label('CurveSphere_Central');actor.set_editor_property('tags',[ue.Name('CurveControlSphereV1')])
    component=actor.static_mesh_component;component.set_static_mesh(mesh);component.set_mobility(ue.ComponentMobility.MOVABLE)
    component.set_collision_profile_name('BlockAll');component.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS)
    material=ue.load_asset('/Game/DroneControlOverview/Materials/M_DCO_Red')
    if material:component.set_material(0,material)
    actor.set_actor_scale3d(ue.Vector(.8,.8,.8))
    # Read the actual default pawn bounds separately from planning-envelope assumptions.
    pawn_class=ue.load_class(None,'/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn_C')
    pawn=ue.EditorLevelLibrary.spawn_actor_from_class(pawn_class,ue.Vector(0,0,1000))
    components=[]
    for c in pawn.get_components_by_class(ue.PrimitiveComponent):
        components.append(dict(name=c.get_name(),type=c.get_class().get_name(),collision_enabled=str(c.get_collision_enabled())))
    origin,extent=pawn.get_actor_bounds(False)
    report['pawn_actor_bounds_extent_cm']=[extent.x,extent.y,extent.z]
    report['pawn_primitive_components']=components
    ue.EditorLevelLibrary.destroy_actor(pawn)
    if not ue.EditorLevelLibrary.save_current_level():raise RuntimeError('03 map save failed')
    asset=json.loads((ROOT/'reports/curve_sphere_asset_build_v1.json').read_text())
    report.update(status='PASS',actor_name=actor.get_name(),collision_radii_cm=asset['collision_radii_cm'],
        installed_map_sha256=sha(source),stored_position_ue_cm=[100000,100000,-50000],
        change='one reserved off-field analytic collision sphere; existing actors and dual-target entry preserved')
except Exception:report.update(status='FAIL',traceback=traceback.format_exc())
out.write_text(json.dumps(report,indent=2),encoding='utf-8');ue.log('CURVE_SPHERE_INSTALL_'+report['status'])

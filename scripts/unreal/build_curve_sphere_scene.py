"""Append a verified sphere to an independent copy of the 03 overview map."""
import hashlib,json,traceback
from pathlib import Path
import unreal as ue
ROOT=Path('D:/forAirsim')
SRC='/Game/DroneControlOverview/Maps/DroneControlOverview'
DST='/Game/CurveControl/Maps/CurveSphereOverview_v1'
content=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content'
source=content/'DroneControlOverview/Maps/DroneControlOverview.umap'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(status='RUNNING',source_map=SRC,map=DST,flight_commands=0,source_sha256=sha(source))
try:
    if ue.EditorAssetLibrary.does_asset_exist(DST):raise RuntimeError('refuse to overwrite existing research map')
    if not ue.EditorLevelLibrary.new_level_from_template(DST,SRC):raise RuntimeError('overview map copy failed')
    mesh=ue.load_asset('/Game/CurveControl/Meshes/SM_CurveControlSphere_v1')
    if not mesh:raise RuntimeError('inspected sphere missing')
    actor=ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor,ue.Vector(400,0,500))
    actor.set_actor_label('CurveSphere_Central')
    actor.set_editor_property('tags',[ue.Name('CurveControlSphereV1')])
    component=actor.static_mesh_component;component.set_static_mesh(mesh)
    component.set_mobility(ue.ComponentMobility.MOVABLE)
    component.set_collision_profile_name('BlockAll')
    component.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS)
    material=ue.load_asset('/Game/DroneControlOverview/Materials/M_DCO_Red')
    if material:component.set_material(0,material)
    actor.set_actor_scale3d(ue.Vector(.8,.8,.8))
    if not ue.EditorLevelLibrary.save_current_level():raise RuntimeError('research map save failed')
    if sha(source)!=report['source_sha256']:raise RuntimeError('source overview changed')
    asset=json.loads((ROOT/'reports/curve_sphere_asset_build_v1.json').read_text())
    report.update(status='PASS',actor_name=actor.get_name(),actor_label=actor.get_actor_label(),
        collision_radii_cm=asset['collision_radii_cm'],mesh_asset='SM_CurveControlSphere_v1',
        map_sha256=sha(content/'CurveControl/Maps/CurveSphereOverview_v1.umap'),source_unchanged=True)
except Exception:report.update(status='FAIL',traceback=traceback.format_exc())
(ROOT/'reports/curve_sphere_scene_build_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
ue.log('CURVE_SPHERE_SCENE_'+report['status'])

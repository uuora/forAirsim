"""Create one uniquely named, metrically inspected runtime obstacle mesh."""
import hashlib,json,traceback
from pathlib import Path
import unreal as ue

ROOT=Path('D:/forAirsim')
DEST='/Game/CurveControl/Meshes/SM_CurveControlSphere_v1'
report=dict(status='RUNNING',destination=DEST,source='/Engine/BasicShapes/Sphere',flight_commands=0,map_mutations=False)
try:
    mesh=ue.load_asset(DEST)
    if mesh:
        if ue.EditorAssetLibrary.get_metadata_tag(mesh,'CurveControlOwner')!='sphere_v1':
            raise RuntimeError('refuse to alter unowned asset')
    else:
        mesh=ue.EditorAssetLibrary.duplicate_asset(report['source'],DEST)
        if not mesh:raise RuntimeError('sphere duplication failed')
        ue.EditorAssetLibrary.set_metadata_tag(mesh,'CurveControlOwner','sphere_v1')
    ue.EditorStaticMeshLibrary.remove_collisions(mesh)
    ue.EditorStaticMeshLibrary.add_simple_collisions(mesh,ue.ScriptingCollisionShapeType.SPHERE)
    bounds=mesh.get_bounds()
    extent=[float(bounds.box_extent.x),float(bounds.box_extent.y),float(bounds.box_extent.z)]
    if any(abs(x-50.)>.001 for x in extent):raise RuntimeError('unit sphere must be one metre in diameter')
    body=mesh.get_editor_property('body_setup')
    primitives=body.get_editor_property('agg_geom').get_editor_property('sphere_elems')
    radii=[float(s.get_editor_property('radius')) for s in primitives]
    if len(radii)!=1 or abs(radii[0]-50.)>.01:raise RuntimeError('explicit 50cm collision sphere required')
    if not ue.EditorAssetLibrary.save_loaded_asset(mesh):raise RuntimeError('asset save failed')
    report.update(status='PASS',visual_extent_cm=extent,collision_radii_cm=radii,
                  simple_collision_count=ue.EditorStaticMeshLibrary.get_simple_collision_count(mesh))
except Exception:
    report.update(status='FAIL',traceback=traceback.format_exc())
(ROOT/'reports/curve_sphere_asset_build_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
ue.log('CURVE_SPHERE_ASSET_'+report['status'])

"""Read-only UE4 asset inventory for the enclosed ground preview."""
import json
from pathlib import Path
import traceback
import unreal as ue

ROOT=Path('D:/forAirsim')
OUT=ROOT/'logs/enclosed_lab_asset_inventory_v2_20261008.json'
report={'status':'RUNNING','asset_mutations':False,'map_mutations':False,'flight_commands':False}
try:
    registry=ue.AssetRegistryHelpers.get_asset_registry()
    report['project_static_meshes']=[]
    for asset in registry.get_assets_by_path('/Game',recursive=True):
        if str(asset.asset_class)=='StaticMesh':
            report['project_static_meshes'].append(str(asset.object_path))
    report['plugin_meshes']=[]
    for asset in registry.get_assets_by_path('/AirSim',recursive=True):
        if str(asset.asset_class) in ('StaticMesh','SkeletalMesh','Blueprint'):
            report['plugin_meshes'].append({'path':str(asset.object_path),'class':str(asset.asset_class)})
    report['static_mesh_library_methods']=[x for x in dir(ue.EditorStaticMeshLibrary) if 'collision' in x or 'lod' in x or 'vertex' in x]
    report['primitives']=[]
    for name in ('Cube','Cylinder','Sphere','Cone','Plane'):
        path='/Engine/BasicShapes/'+name
        mesh=ue.load_asset(path)
        row={'path':path,'loaded':bool(mesh)}
        if mesh:
            bounds=mesh.get_bounds()
            row.update(origin_cm=[bounds.origin.x,bounds.origin.y,bounds.origin.z],
                extent_cm=[bounds.box_extent.x,bounds.box_extent.y,bounds.box_extent.z],
                simple_collision_count=ue.EditorStaticMeshLibrary.get_simple_collision_count(mesh),
                convex_collision_count=ue.EditorStaticMeshLibrary.get_convex_collision_count(mesh))
        report['primitives'].append(row)
    report['collision_shape_options']=[x for x in dir(ue.ScriptingCollisionShapeType) if x.isupper()]
    report['status']='PASS'
except Exception:
    report.update(status='FAIL',error=traceback.format_exc())
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
ue.log('ENCLOSED_LAB_INVENTORY_'+report['status'])

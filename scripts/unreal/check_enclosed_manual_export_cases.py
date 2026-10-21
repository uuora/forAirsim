"""In-memory rotated/duplicated obstacle and invalid sphere cases. Never saves a map."""
import importlib.util
import json
import traceback
from datetime import datetime
from pathlib import Path
import unreal as ue

ROOT=Path('D:/forAirsim')
spec=importlib.util.spec_from_file_location('manual_geometry',ROOT/'scripts/unreal/enclosed_lab_manual.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
receipt=dict(status='RUNNING', saved_map_mutations=False)
try:
    if not ue.EditorLevelLibrary.load_level(module.WORK_MAP):
        raise RuntimeError('LOAD_FAILED')
    actors=ue.EditorLevelLibrary.get_all_level_actors()
    box=next(a for a in actors if a.get_actor_label()=='EL1_LowBlock')
    box.set_actor_rotation(ue.Rotator(pitch=17,yaw=35,roll=11),False)
    duplicate=ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor,ue.Vector(1200,1200,60))
    duplicate.set_actor_label('EL1_LowBlock')
    c=duplicate.static_mesh_component
    c.set_static_mesh(box.static_mesh_component.get_editor_property('static_mesh'))
    c.set_material(0,box.static_mesh_component.get_material(0))
    duplicate.set_actor_scale3d(ue.Vector(4,2,1.2))
    c.set_collision_profile_name('BlockAll')
    c.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS)
    module.OUT=ROOT/'reports'/('enclosed_manual_export_test_rotated_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    module.audit(load_saved=False)
    receipt['rotated_and_duplicate_status']=module.report['status']
    if module.report['status']!='PASS_IN_MEMORY_TEST_GEOMETRY':
        raise RuntimeError('ROTATED_EXPORT_FAILED: '+json.dumps(module.report['problems']))
    geometry=json.loads((module.OUT/'planning_geometry.json').read_text(encoding='utf-8'))
    boxes=[p for p in geometry['primitives'] if p['display_label']=='EL1_LowBlock']
    if len(boxes)!=2 or len({p['id'] for p in boxes})!=2 or not all(p['shape']=='obb' for p in boxes):
        raise RuntimeError('DUPLICATE_POPULATION_NOT_PRESERVED')
    sphere=next(a for a in actors if a.get_actor_label()=='EL1_Balloon_Target')
    sphere.set_actor_scale3d(ue.Vector(2,2,3))
    module.OUT=ROOT/'reports'/('enclosed_manual_export_test_invalid_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    module.audit(load_saved=False)
    receipt['nonuniform_sphere_status']=module.report['status']
    if module.report['status']!='FAIL_GEOMETRY_REQUIRES_REVIEW' or not any('uniform' in p['reason'] for p in module.report['problems']):
        raise RuntimeError('NONUNIFORM_SPHERE_WAS_NOT_REJECTED')
    receipt['status']='PASS_EXPORT_BEHAVIOR_CASES'
except Exception:
    receipt.update(status='FAIL',error=traceback.format_exc())
finally:
    module.save(ROOT/'logs/enclosed_manual_export_cases.json',receipt)
    ue.log('MANUAL_EXPORT_CASES_'+receipt['status'])

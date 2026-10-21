"""Prepare a separate editor work copy, or audit its saved geometry. No flight."""
import csv
import hashlib
import json
import math
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
import unreal as ue

ROOT = Path('D:/forAirsim')
BASE = '/Game/EnclosedUAVLabV1'
WORK = '/Game/EnclosedUAVLabManual'
SOURCE_MAP = BASE + '/Maps/EnclosedUAVLab'
WORK_MAP = WORK + '/Maps/EnclosedUAVLab_Manual'
MODE = os.environ.get('ENCLOSED_LAB_MANUAL_MODE', 'prepare')
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT = ROOT / 'reports' / ('enclosed_lab_manual_' + STAMP)
LATEST = ROOT / 'logs' / 'enclosed_lab_manual_latest.json'
report = dict(status='RUNNING', mode=MODE, map=WORK_MAP, flight_validation='NOT_RUN', flight_commands=False)


def values(v):
    return [float(v.x), float(v.y), float(v.z)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def clone_assets():
    """Duplicate independently and explicitly rebind; DuplicateDirectory doesn't remap references."""
    assets = ue.EditorAssetLibrary.list_assets(BASE, recursive=True, include_folder=False)
    copies = {}
    for path in assets:
        old = ue.load_asset(path)
        if isinstance(old, (ue.StaticMesh, ue.Material)):
            destination = old.get_path_name().split('.')[0].replace(BASE, WORK, 1)
            new = ue.load_asset(destination) or ue.EditorAssetLibrary.duplicate_asset(path, destination)
            if new is None:
                raise RuntimeError('ASSET_COPY_FAILED: ' + path)
            copies[old.get_path_name()] = new
    for new in copies.values():
        if isinstance(new, ue.StaticMesh):
            for i in range(new.get_num_sections(0)):
                old_mat = new.get_material(i)
                if old_mat and old_mat.get_path_name() in copies:
                    new.set_material(i, copies[old_mat.get_path_name()])
    return copies


def prepare():
    created = not ue.EditorAssetLibrary.does_asset_exist(WORK_MAP)
    if created:
        copies = clone_assets()
        if not ue.EditorLevelLibrary.new_level_from_template(WORK_MAP, SOURCE_MAP):
            raise RuntimeError('WORK_COPY_MAP_FAILED')
        for actor in ue.EditorLevelLibrary.get_all_level_actors():
            if isinstance(actor, ue.StaticMeshActor):
                c = actor.static_mesh_component
                old_mesh = c.get_editor_property('static_mesh')
                if old_mesh and old_mesh.get_path_name() in copies:
                    c.set_static_mesh(copies[old_mesh.get_path_name()])
                for i in range(c.get_num_materials()):
                    m = c.get_material(i)
                    if m and m.get_path_name() in copies:
                        c.set_material(i, copies[m.get_path_name()])
        if not ue.EditorLevelLibrary.save_current_level():
            raise RuntimeError('WORK_COPY_SAVE_FAILED')
        ue.EditorAssetLibrary.save_directory(WORK, only_if_is_dirty=False, recursive=True)
    elif not ue.EditorLevelLibrary.load_level(WORK_MAP):
        raise RuntimeError('EXISTING_WORK_MAP_LOAD_FAILED')
    config = ROOT / 'configs' / 'enclosed_uav_lab_manual.json'
    if not config.exists():
        scene = json.loads((ROOT / 'configs' / 'enclosed_uav_lab_v1.json').read_text(encoding='utf-8'))
        scene.update(scene_id='enclosed_uav_lab_manual', unreal_map=WORK_MAP,
                     edit_authority='Parameter build replaces this working map; editor edits must be exported separately.')
        save(config, scene)
    settings=ROOT/'configs/enclosed_uav_lab_manual_settings.json'
    if not settings.exists():
        data=json.loads((ROOT/'configs/enclosed_uav_lab_preview_settings_v1.json').read_text(encoding='utf-8'))
        data['ApiServerPort']=41454
        save(settings,data)
    ue.EditorLevelLibrary.set_level_viewport_camera_info(
        ue.Vector(-1600, -3300, 3200), ue.Rotator(pitch=-28, yaw=39, roll=0))
    report.update(status='PASS_WORK_COPY_READY', created_new_copy=created, config=str(config), settings=str(settings),
                  existing_editor_edits_preserved=not created)


def trace(world, start, end, ignore):
    hit = ue.SystemLibrary.line_trace_single(world, ue.Vector(*[100*x for x in start]),
        ue.Vector(*[100*x for x in end]), ue.TraceTypeQuery.TRACE_TYPE_QUERY1,
        False, ignore, ue.DrawDebugTrace.NONE, True)
    if hit is None:
        return dict(blocking=False)
    fields = hit.to_tuple()
    if len(fields) != 17:
        raise RuntimeError('UNEXPECTED_NATIVE_HIT_RESULT_LAYOUT')
    return dict(blocking=bool(fields[0]), impact_xyz_m=[x/100 for x in values(fields[5])],
                actor=fields[9].get_name() if fields[9] else None)


def audit(load_saved=True):
    if load_saved and not ue.EditorLevelLibrary.load_level(WORK_MAP):
        raise RuntimeError('Save and close the editor first; working map unavailable.')
    actors = ue.EditorLevelLibrary.get_all_level_actors()
    world = ue.EditorLevelLibrary.get_editor_world()
    source = json.loads((ROOT / 'configs/enclosed_uav_lab_v1.json').read_text(encoding='utf-8'))
    known = {p['id']: p for p in source['parts']}
    rows, primitives, problems = [], [], []
    allowed = {BASE+'/Meshes/SM_Box.SM_Box': 'box', WORK+'/Meshes/SM_Box.SM_Box': 'box',
               BASE+'/Meshes/SM_Sphere.SM_Sphere': 'sphere', WORK+'/Meshes/SM_Sphere.SM_Sphere': 'sphere'}
    for a in actors:
        if not isinstance(a, ue.StaticMeshActor):
            if isinstance(a, ue.PlayerStart):
                continue  # Editor-only start indicator; not an experiment obstacle.
            for c in a.get_components_by_class(ue.PrimitiveComponent):
                if c.get_collision_enabled() != ue.CollisionEnabled.NO_COLLISION:
                    problems.append(dict(actor=a.get_name(), reason='Unmapped colliding component; convert to audited static primitive or extend exporter.'))
            continue
        c = a.static_mesh_component
        mesh = c.get_editor_property('static_mesh')
        collides = c.get_collision_enabled() != ue.CollisionEnabled.NO_COLLISION
        if mesh is None:
            if collides:
                problems.append(dict(actor=a.get_name(), reason='Enabled collision with no mesh.'))
            continue
        mesh_path = mesh.get_path_name()
        kind = allowed.get(mesh_path)
        label = a.get_actor_label()
        original = known.get(label, {})
        center = [x/100 for x in values(a.get_actor_location())]
        scale = values(a.get_actor_scale3d())
        axes = [values(a.get_actor_forward_vector()), values(a.get_actor_right_vector()), values(a.get_actor_up_vector())]
        origin, extent = a.get_actor_bounds(False)
        actual_center = [x/100 for x in values(origin)]
        actual_extent = [x/100 for x in values(extent)]
        bounds = [[actual_center[i]+s*actual_extent[i] for i in range(3)] for s in [-1, 1]]
        row = dict(id=a.get_name(), display_label=label, source_id=original.get('id'),
                   group=original.get('group', 'manual_addition'), role=original.get('role', 'obstacle'),
                   mesh_path=mesh_path, material_paths=[c.get_material(i).get_path_name() if c.get_material(i) else None for i in range(c.get_num_materials())],
                   center_xyz_m=center, local_size_xyz_m=scale, orientation_axes=axes,
                   rotation_degrees=dict(pitch=a.get_actor_rotation().pitch, yaw=a.get_actor_rotation().yaw, roll=a.get_actor_rotation().roll),
                   visible_aabb_xyz_m=bounds, collision_enabled=str(c.get_collision_enabled()), collides=collides,
                   collision_profile=str(c.get_collision_profile_name()), planner_shape='none', checks=[])
        rows.append(row)
        if not kind:
            if collides:
                problems.append(dict(actor=a.get_name(), reason='Mesh is outside the verified box/sphere asset set: '+mesh_path))
            continue
        if min(scale) <= 0 or not all(math.isfinite(v) for v in center+scale):
            problems.append(dict(actor=a.get_name(), reason='Require finite positive scale.'))
            continue
        expected_extent = [sum(abs(axes[j][i])*scale[j]/2 for j in range(3)) for i in range(3)] if kind=='box' else [scale[0]/2]*3
        if kind == 'sphere' and max(scale)-min(scale) > .00001:
            problems.append(dict(actor=a.get_name(), reason='Sphere requires uniform XYZ scale; ellipsoid collision is not exported.'))
            continue
        # A sphere's tessellated visible bounds can vary with rotation; analytic physics is stated explicitly.
        error = max(abs(center[i]-actual_center[i]) for i in range(3))
        if kind=='box':
            error = max(error, max(abs(expected_extent[i]-actual_extent[i]) for i in range(3)))
        row['visible_bounds_error_m'] = error
        if error > .005:
            problems.append(dict(actor=a.get_name(), reason='Mesh/component transform does not match actor geometry.'))
            continue
        if collides:
            if a.get_editor_property('hidden') or c.get_editor_property('hidden_in_game') or not c.get_editor_property('visible'):
                problems.append(dict(actor=a.get_name(), reason='Invisible colliding obstacle.'))
            if c.get_collision_enabled()!=ue.CollisionEnabled.QUERY_AND_PHYSICS:
                problems.append(dict(actor=a.get_name(), reason='Require both query and physics collision.'))
            if str(c.get_collision_profile_name()) != 'BlockAll':
                problems.append(dict(actor=a.get_name(), reason='Require BlockAll for this audited obstacle set.'))
            if ue.EditorStaticMeshLibrary.get_simple_collision_count(mesh)!=1 or ue.EditorStaticMeshLibrary.get_convex_collision_count(mesh)!=0:
                problems.append(dict(actor=a.get_name(), reason='Primitive collision count changed.'))
            if ue.EditorStaticMeshLibrary.get_collision_complexity(mesh)!=ue.CollisionTraceFlag.CTF_USE_DEFAULT:
                problems.append(dict(actor=a.get_name(), reason='Collision complexity changed from verified default.'))
        ignored = [other for other in actors if other != a]
        for axis in range(3):
            half = scale[axis]/2
            for sign in [-1, 1]:
                d = axes[axis]
                begin = [center[i]+sign*d[i]*(half+.25) for i in range(3)]
                end = [center[i]-sign*d[i]*(half+.25) for i in range(3)]
                hit = trace(world, begin, end, ignored)
                if hit['blocking'] != collides or (collides and hit.get('actor') != a.get_name()):
                    problems.append(dict(actor=a.get_name(), reason='Isolated axial ray mismatch.', ray=hit))
                elif collides:
                    face = [center[i]+sign*d[i]*half for i in range(3)]
                    distance = math.sqrt(sum((hit['impact_xyz_m'][i]-face[i])**2 for i in range(3)))
                    if distance > .015:
                        problems.append(dict(actor=a.get_name(), reason='Collision surface differs by more than 15 mm.', error_m=distance))
                    hit['surface_error_m'] = distance
                row['checks'].append(dict(axis=axis, sign=sign, ray=hit))
            d, perp = axes[axis], axes[(axis+1)%3]
            begin = [center[i]-d[i]*(half+.25)+perp[i]*(scale[(axis+1)%3]/2+.05) for i in range(3)]
            end = [center[i]+d[i]*(half+.25)+perp[i]*(scale[(axis+1)%3]/2+.05) for i in range(3)]
            hit = trace(world, begin, end, ignored)
            if hit['blocking']:
                problems.append(dict(actor=a.get_name(), reason='Collision extends past the expected primitive.', ray=hit))
            row['checks'].append(dict(outside_axis=axis, ray=hit))
        if collides:
            p = dict(id=a.get_name(), display_label=label, center_xyz_m=center, group=row['group'], role=row['role'])
            if kind == 'box':
                row['planner_shape']='obb'
                p.update(shape='obb', local_size_xyz_m=scale, orientation_axes=axes,
                         conservative_aabb_xyz_m=[[center[i]+s*expected_extent[i] for i in range(3)] for s in [-1,1]])
            else:
                row['planner_shape']='sphere'
                p.update(shape='sphere', radius_m=scale[0]/2)
            primitives.append(p)
    if not rows:
        problems.append(dict(reason='No visible mesh actors found.'))
    # All recorded rows use actual unique native actor names; duplicated display labels cannot merge obstacles.
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'scene_geometry.json',dict(schema_version=2, coordinate_system='metres, X forward, Y right, Z up',
         map=WORK_MAP, parts=rows, flight_authorized=False))
    status = ('PASS_SAVED_EDITOR_GEOMETRY' if load_saved else 'PASS_IN_MEMORY_TEST_GEOMETRY') if not problems else 'FAIL_GEOMETRY_REQUIRES_REVIEW'
    save(OUT/'planning_geometry.json', dict(status=status, schema_version=2, primitives=primitives,
         units='m', coordinate_system='engineering Z up', clearance_inflation_m=0,
         controller_integration='NOT_CONNECTED', rotation_contract='OBB axes are unit world vectors for local X, Y, Z; do not replace an open assembly by its enclosing box.'))
    with (OUT/'geometry_correspondence.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['native_actor_id','display_label','source_id','mesh','collision_enabled','planner_shape','center_xyz_m','local_size_xyz_m','rotation_degrees'])
        for r in rows:
            writer.writerow([r[k] if isinstance(r[k], str) or r[k] is None else json.dumps(r[k]) for k in
                             ['id','display_label','source_id','mesh_path','collision_enabled','planner_shape','center_xyz_m','local_size_xyz_m','rotation_degrees']])
    report.update(status=status, output=str(OUT), parts_checked=len(rows), planner_primitives=len(primitives), problems=problems,
                  geometry_source='saved_editor_map' if load_saved else 'unsaved_in_memory_test',
                  limitations=['Editor geometry only; no runtime capture or flight validation for manual revisions.',
                    'Six axial and three outside ray samples per primitive are bounded checks, not exhaustive contact tests.',
                    'Analytic sphere collision/planning differs from the tessellated visible sphere surface.',
                    'UAV spawn settings and controller are not regenerated; moving the start platform requires separate alignment.'])
    save(OUT/'geometry_check.json', report)
    folder=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual'
    disk_assets=[p for p in folder.rglob('*') if p.suffix in ('.uasset','.umap')]
    if load_saved:
        for p in disk_assets:
            destination=OUT/'unreal_asset_snapshot'/p.relative_to(ROOT/'external/AirSim/Unreal/Environments/Blocks/Content')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,destination)
            if sha(destination)!=sha(p):
                raise RuntimeError('ASSET_SNAPSHOT_CHANGED_DURING_EXPORT')
        shutil.copyfile(Path(__file__),OUT/'exporter_source.py')
    save(OUT/'provenance.json',dict(map=WORK_MAP, status=status, exported_at=STAMP,
         geometry_source=report['geometry_source'], disk_asset_hashes_are_geometry_source=load_saved,
         source_script_sha256=sha(Path(__file__)),
         unreal_asset_sha256={str(p.relative_to(ROOT)):sha(p) for p in disk_assets},
         output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},
         AI_assistance='AI assisted export and bounded geometry checks; human scene acceptance pending.', flight_validation='NOT_RUN'))
    if load_saved:
        save(ROOT/'reports/enclosed_lab_manual_geometry_latest.json',dict(status=status,output=str(OUT),report=str(OUT/'geometry_check.json')))


def main():
    try:
        if MODE == 'prepare':
            prepare()
        elif MODE == 'audit':
            audit()
        else:
            raise RuntimeError('Unknown mode: '+MODE)
    except Exception:
        report.update(status='FAIL', error=traceback.format_exc())
    finally:
        receipt=ROOT/'logs'/('enclosed_lab_manual_'+MODE+'_'+STAMP+'.json')
        save(receipt, report)
        save(LATEST, dict(status=report['status'], mode=MODE, report=str(receipt), output=report.get('output')))
        ue.log('ENCLOSED_LAB_MANUAL_'+report['status'])


if __name__=='__main__':
    main()

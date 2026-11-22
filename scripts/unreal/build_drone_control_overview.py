"""Build the isolated UE 4.27 overview map; never launch or command a drone.

The collision floor is identical to the lightweight control map. The visible
surface, grid and targets are decorative and never enter collision queries.
An API SceneCapture camera is supplied by AirSim settings, not by this builder.
"""
import hashlib
import json
import math
import traceback
from pathlib import Path

import unreal as ue


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/dual_target_control.json'
BASE = '/Game/DroneControlOverview'
MAP = BASE + '/Maps/DroneControlOverview'
BUILDER = 'DroneControlOverview'
VERSION = '1'
OWNED_TAG = 'DCO_BuilderOwned'
VERSION_TAG = 'DCO_BuilderVersion_' + VERSION
METADATA_KEY = 'DCO_Builder'
RECEIPT = ROOT / 'logs/dual_target_control_overview_build.json'

receipt = dict(map=MAP, builder=BUILDER, builder_version=VERSION,
               status='BUILDING', flight_commands=False)
asset_tools = ue.AssetToolsHelpers.get_asset_tools()
mel = ue.MaterialEditingLibrary


def tags(actor):
    return {str(tag) for tag in actor.get_editor_property('tags')}


def own(actor, label):
    actor.set_actor_label(label)
    actor.set_editor_property('tags', [ue.Name(OWNED_TAG), ue.Name(VERSION_TAG)])
    actor.set_folder_path(ue.Name('DroneControlOverview'))
    return actor


def expression(material, kind, **properties):
    node = mel.create_material_expression(material, kind)
    for key, value in properties.items():
        node.set_editor_property(key, value)
    return node


def unlit_material(name, rgb):
    path = BASE + '/Materials/M_DCO_' + name
    material = ue.load_asset(path)
    if material:
        owner = ue.EditorAssetLibrary.get_metadata_tag(material, METADATA_KEY)
        if owner != BUILDER + ':' + VERSION:
            raise RuntimeError('Refusing to alter an unowned material: ' + path)
    else:
        material = asset_tools.create_asset('M_DCO_' + name, BASE + '/Materials',
                                            ue.Material, ue.MaterialFactoryNew())
        if not material:
            raise RuntimeError('Cannot create material: ' + path)
        ue.EditorAssetLibrary.set_metadata_tag(material, METADATA_KEY, BUILDER + ':' + VERSION)
    material.set_editor_property('shading_model', ue.MaterialShadingModel.MSM_UNLIT)
    mel.delete_all_material_expressions(material)
    color = expression(material, ue.MaterialExpressionConstant3Vector,
                       constant=ue.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    mel.connect_material_property(color, '', ue.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(material)
    if not ue.EditorAssetLibrary.save_loaded_asset(material):
        raise RuntimeError('Cannot save material: ' + path)
    return material


def primitive(label, center_m, size_m, material=None, collides=False, sphere=False):
    actor = own(ue.EditorLevelLibrary.spawn_actor_from_class(
        ue.StaticMeshActor, ue.Vector(*[100.0 * v for v in center_m])), label)
    component = actor.static_mesh_component
    component.set_static_mesh(ue.load_asset('/Engine/BasicShapes/' + ('Sphere' if sphere else 'Cube')))
    component.set_mobility(ue.ComponentMobility.STATIC)
    component.set_collision_profile_name('BlockAll' if collides else 'NoCollision')
    component.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS if collides
                                    else ue.CollisionEnabled.NO_COLLISION)
    component.set_editor_property('cast_shadow', False)
    if material:
        component.set_material(0, material)
    actor.set_actor_scale3d(ue.Vector(*size_m))
    return actor


def label(name, value, x_m, y_m, color=(42, 52, 64)):
    actor = own(ue.EditorLevelLibrary.spawn_actor_from_class(
        ue.TextRenderActor, ue.Vector(100.0 * x_m, 100.0 * y_m, 6.0),
        ue.Rotator(pitch=90, yaw=0, roll=0)), 'DC_OverviewLabel_' + name)
    component = actor.get_component_by_class(ue.TextRenderComponent)
    component.set_text(value)
    component.set_world_size(150.0)
    component.set_text_render_color(ue.Color(color[0], color[1], color[2], 255))
    component.set_horizontal_alignment(ue.HorizTextAligment.EHTA_CENTER)
    component.set_collision_enabled(ue.CollisionEnabled.NO_COLLISION)
    return actor


def bounded_grid_values(low, high, spacing):
    count = int(math.floor(high / spacing) - math.ceil(low / spacing)) + 1
    if count > 100:
        raise RuntimeError('Too many grid lines; increase overview.grid_spacing_m')
    return [index * spacing for index in range(int(math.ceil(low / spacing)),
                                              int(math.floor(high / spacing)) + 1)]


try:
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig')) if CONFIG.exists() else {}
    control = config.get('control', {})
    overview = config.get('overview', {})
    xmin = float(control.get('x_min', -50.0))
    xmax = float(control.get('x_max', 50.0))
    ymin = float(control.get('y_min', -50.0))
    ymax = float(control.get('y_max', 50.0))
    spacing = float(overview.get('grid_spacing_m', 10.0))
    padding = float(overview.get('floor_padding_m', 10.0))
    bounds = [xmin, xmax, ymin, ymax, spacing, padding]
    if not all(math.isfinite(value) for value in bounds):
        raise RuntimeError('Non-finite overview geometry')
    if xmin >= xmax or ymin >= ymax or spacing <= 0 or padding < 0:
        raise RuntimeError('Invalid overview geometry')
    if max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + padding >= 500.0:
        raise RuntimeError('Visible field must stay inside the 1000 m collision floor')
    xs = bounded_grid_values(xmin, xmax, spacing)
    ys = bounded_grid_values(ymin, ymax, spacing)
    geometry = dict(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax,
                    grid_spacing_m=spacing, floor_padding_m=padding)
    receipt['geometry'] = geometry
    receipt['geometry_sha256'] = hashlib.sha256(
        json.dumps(geometry, sort_keys=True).encode('utf-8')).hexdigest()

    if ue.EditorAssetLibrary.does_asset_exist(MAP):
        if not ue.EditorLevelLibrary.load_level(MAP):
            raise RuntimeError('Cannot load ' + MAP)
        existing = ue.EditorLevelLibrary.get_all_level_actors()
        marker = next((a for a in existing if a.get_actor_label() == 'DC_OverviewMarker'), None)
        if not marker or OWNED_TAG not in tags(marker) or VERSION_TAG not in tags(marker):
            raise RuntimeError('Existing map lacks this builder version; refusing to replace it')
        # Only actors minted by this builder are rebuilt; user-added actors survive.
        for actor in existing:
            if OWNED_TAG in tags(actor):
                ue.EditorLevelLibrary.destroy_actor(actor)
    elif not ue.EditorLevelLibrary.new_level(MAP):
        raise RuntimeError('Cannot create ' + MAP)

    world = ue.EditorLevelLibrary.get_editor_world()
    mode = ue.load_class(None, '/Script/AirSim.AirSimGameMode')
    if not mode:
        raise RuntimeError('AirSim game mode unavailable')
    world.get_world_settings().set_editor_property('default_game_mode', mode)
    own(ue.EditorLevelLibrary.spawn_actor_from_class(ue.Actor, ue.Vector(0, 0, 0)),
        'DC_OverviewMarker')
    own(ue.EditorLevelLibrary.spawn_actor_from_class(ue.PlayerStart, ue.Vector(0, 0, 100)),
        'DC_PlayerStart')

    materials = {name: unlit_material(name, rgb) for name, rgb in {
        'Floor': (0.68, 0.71, 0.74), 'Grid': (0.16, 0.19, 0.22),
        'Axis': (0.08, 0.12, 0.18), 'Boundary': (0.015, 0.22, 0.40),
        'Red': (1.0, 0.015, 0.01), 'Blue': (0.015, 0.16, 1.0),
    }.items()}

    floor = primitive('DC_CollisionFloor', (0, 0, -0.5), (1000, 1000, 1), collides=True)
    floor.set_actor_hidden_in_game(True)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    width, depth = xmax - xmin, ymax - ymin
    primitive('DC_VisibleFloor', (xc, yc, -0.005),
              (width + 2 * padding, depth + 2 * padding, 0.01), materials['Floor'])
    for index, x in enumerate(xs):
        primitive('DC_GridX_' + str(index), (x, yc, 0.015),
                  (0.07, depth, 0.01), materials['Axis' if abs(x) < 1e-6 else 'Grid'])
    for index, y in enumerate(ys):
        primitive('DC_GridY_' + str(index), (xc, y, 0.025),
                  (width, 0.07, 0.01), materials['Axis' if abs(y) < 1e-6 else 'Grid'])
    for name, x in [('Left', xmin), ('Right', xmax)]:
        primitive('DC_Boundary_' + name, (x, yc, 0.04),
                  (0.20, depth + 0.20, 0.02), materials['Boundary'])
    for name, y in [('Near', ymin), ('Far', ymax)]:
        primitive('DC_Boundary_' + name, (xc, y, 0.04),
                  (width + 0.20, 0.20, 0.02), materials['Boundary'])
    label('X', 'X +', xmax - 5.0, yc + 2.0)
    label('Y', 'Y +', xc + 3.0, ymax - 5.0)
    label('Field', '{:g} x {:g} m / grid {:g} m'.format(width, depth, spacing),
          xc, ymin - min(padding * 0.6, 6.0))

    for color in ('Red', 'Blue'):
        target = primitive('DC_Target' + color, (0, 0, -500), (1.2, 1.2, 1.2),
                           materials[color], sphere=True)
        # Runtime moves these targets; neither target ever creates an obstacle.
        target.static_mesh_component.set_mobility(ue.ComponentMobility.MOVABLE)

    sun = own(ue.EditorLevelLibrary.spawn_actor_from_class(
        ue.DirectionalLight, ue.Vector(0, 0, 1800),
        ue.Rotator(pitch=-48, yaw=-38, roll=0)), 'DC_OverviewSun')
    sun.light_component.set_mobility(ue.ComponentMobility.MOVABLE)
    sun.light_component.set_editor_property('intensity', 4.0)
    sun.light_component.set_atmosphere_sun_light(True)
    own(ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyAtmosphere, ue.Vector(0, 0, 0)),
        'DC_OverviewAtmosphere')
    sky = own(ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyLight, ue.Vector(0, 0, 1000)),
              'DC_OverviewSkyLight')
    sky.light_component.set_mobility(ue.ComponentMobility.MOVABLE)
    sky.light_component.set_editor_property('intensity', 1.0)
    sky.light_component.set_editor_property('real_time_capture', True)

    if not ue.EditorLevelLibrary.save_current_level():
        raise RuntimeError('Cannot save ' + MAP)
    actors = ue.EditorLevelLibrary.get_all_level_actors()
    labels = [actor.get_actor_label() for actor in actors]
    required = {'DC_CollisionFloor', 'DC_PlayerStart', 'DC_TargetRed', 'DC_TargetBlue',
                'DC_VisibleFloor', 'DC_OverviewMarker'}
    if not required.issubset(set(labels)):
        raise RuntimeError('Required overview actors absent')
    receipt.update(status='PASS', actor_labels=labels, actor_count=len(actors),
                   collision_floor_cm=dict(center=[0, 0, -50], scale=[1000, 1000, 1]),
                   target_radius_m=0.6, grid_line_count=len(xs) + len(ys),
                   external_camera_required=True)
except Exception:
    receipt.update(status='FAIL', error=traceback.format_exc())
finally:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    ue.log(str(receipt))

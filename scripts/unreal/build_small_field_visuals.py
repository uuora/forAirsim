"""Build scaled render materials from the installed AirSim pawn materials."""
import json
from pathlib import Path

import unreal as ue


ROOT = Path(__file__).resolve().parents[2]
DEST = '/Game/SmallFieldVisuals'
SCALE = json.loads((ROOT / 'configs/small_field.json').read_text())['ue_visualization']['drone_scale']
PAWN = '/AirSim/Blueprints/BP_FlyingPawn'
RECIPE = ROOT / '.runtime/small_field_visuals.json'


def make_material(source, name):
    duplicate = ue.EditorAssetLibrary.duplicate_asset(source.get_path_name(), f'{DEST}/{name}')
    if not duplicate:
        raise RuntimeError(f'Cannot duplicate {source.get_path_name()}')
    if isinstance(duplicate, ue.MaterialInstanceConstant):
        parent = duplicate.get_editor_property('parent')
        ue.MaterialEditingLibrary.set_material_instance_parent(duplicate, make_material(parent, name + '_Base'))
    elif isinstance(duplicate, ue.Material):
        graph = ue.MaterialEditingLibrary
        pivot = graph.create_material_expression(duplicate, ue.MaterialExpressionActorPositionWS, -720, 620)
        position = graph.create_material_expression(duplicate, ue.MaterialExpressionWorldPosition, -720, 790)
        delta = graph.create_material_expression(duplicate, ue.MaterialExpressionSubtract, -460, 650)
        offset = graph.create_material_expression(duplicate, ue.MaterialExpressionMultiply, -220, 650)
        offset.set_editor_property('const_b', 1 - SCALE)
        connections = (
            graph.connect_material_expressions(pivot, '', delta, 'A'),
            graph.connect_material_expressions(position, '', delta, 'B'),
            graph.connect_material_expressions(delta, '', offset, 'A'),
        )
        if not all(connections):
            raise RuntimeError(f'Cannot connect vertex displacement on {source.get_path_name()}')
        # In UE 4.27 the editor Python enum omits the hidden WPO entry.
        ue.SystemLibrary.execute_console_command(ue.EditorLevelLibrary.get_editor_world(),
            'ke /Script/MaterialEditor.Default__MaterialEditingLibrary '
            f'ConnectMaterialProperty {offset.get_path_name()} "" MP_WorldPositionOffset')
        graph.recompile_material(duplicate)
    else:
        raise TypeError(f'Unsupported material {source.get_path_name()}')
    ue.EditorAssetLibrary.save_loaded_asset(duplicate, only_if_is_dirty=False)
    return duplicate


def run():
    actor = ue.EditorLevelLibrary.spawn_actor_from_class(
        ue.EditorAssetLibrary.load_blueprint_class(PAWN), ue.Vector(0, 0, 1000))
    recipes = {}
    materials = {}
    try:
        for mesh in actor.get_components_by_class(ue.StaticMeshComponent):
            slots = []
            for index in range(mesh.get_num_materials()):
                source = mesh.get_material(index)
                path = source.get_path_name()
                if path not in materials:
                    name = source.get_name() + '_Scale' + str(SCALE).replace('.', 'p')
                    asset = ue.EditorAssetLibrary.load_asset(f'{DEST}/{name}') or make_material(source, name)
                    materials[path] = asset.get_path_name()
                slots.append({'original': path, 'scaled': materials[path]})
            recipes[mesh.get_name()] = slots
    finally:
        ue.EditorLevelLibrary.destroy_actor(actor)
    receipt = {'scale': SCALE, 'components': recipes}
    RECIPE.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    ue.log('SMALL_FIELD_MATERIALS ' + json.dumps(receipt))


run()

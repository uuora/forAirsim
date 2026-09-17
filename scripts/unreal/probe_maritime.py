import json
from pathlib import Path
import unreal
names = ['MaterialExpressionCustom', 'CustomInput', 'CustomMaterialOutputType', 'SkyAtmosphere', 'SkyLight', 'DirectionalLight', 'ExponentialHeightFog', 'FbxImportUI', 'EditorLevelLibrary']
result = {name: hasattr(unreal, name) for name in names}
result['game_mode'] = str(unreal.load_class(None, '/Script/AirSim.AirSimGameMode'))
result['custom_input'] = str(unreal.CustomInput(input_name='P'))
Path('D:/forAirsim/logs/maritime_unreal_probe.json').write_text(json.dumps(result, indent=2))

"""Machine-only UE executable override; physical/control configs stay shared."""
from copy import deepcopy
import json,os
from pathlib import Path


def apply_machine_config(config,root=None):
    root=Path(root or Path(__file__).resolve().parents[2]);cfg=deepcopy(config)
    local=root/'.runtime/local_machine.json';settings=json.loads(local.read_text(encoding='utf-8-sig')) if local.exists() else {}
    engine=os.environ.get('AIRSIM_UE_ROOT') or settings.get('engine_root')
    editor=str(Path(engine)/'Engine/Binaries/Win64/UE4Editor.exe') if engine else settings.get('airsim_editor')
    if editor and 'airsim' in cfg:cfg['airsim']['editor']=editor
    return cfg

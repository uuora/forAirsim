"""Read-only relocation of recorded workspace paths; never edits evidence."""
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
SECTIONS=('logs','reports','python','scripts','configs','tmp','external','experiments')


def leaf_name(value):return PurePosixPath(str(value).replace('\\','/')).name


def workspace_record_path(value,root=None):
    root=Path(root or ROOT).resolve();text=str(value).replace('\\','/')
    # Resolve against THIS checkout even when an old checkout still exists.
    # Only known project sections are eligible, and traversal is rejected.
    for section in SECTIONS:
        marker='/'+section+'/'
        if text.startswith(section+'/'):suffix=text
        elif marker in text:suffix=text.split(marker,1)[1];suffix=section+'/'+suffix
        else:continue
        target=(root/suffix).resolve()
        if root not in target.parents:raise ValueError('记录路径越出当前工作区')
        return target
    p=Path(text)
    return p if p.is_absolute() else root/p

"""Archive existing geometric results without flight or changes to raw evidence."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from geometric_validation.runtime import filesystem_path


def main():
    package=ROOT/'experiments/geometric_planners_20261106'
    package.mkdir(exist_ok=True)
    archive=package/'evidence.zip'
    if archive.exists():raise FileExistsError('Evidence archive already exists; preserve the published package')
    patterns=('reports/geometric_display_revision_20261106_v1','reports/geometric_environments_20261106_v*',
        'reports/geometric_environment_suite_20261106_160125_208546',
        'reports/geometric_three_planners_20261106_*','logs/geometric_20261106_*')
    source_root=filesystem_path(ROOT)
    paths=sorted({p for pattern in patterns for folder in ROOT.glob(pattern)
        for p in filesystem_path(folder).rglob('*') if p.is_file()})
    manifest={};omitted=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
        for path in paths:
            name=path.relative_to(source_root).as_posix()
            # UE user preferences and transient caches are not experiment evidence.
            if '/Saved/Config/' in name or '/__pycache__/' in name or path.suffix=='.pyc':
                omitted.append(name);continue
            data=path.read_bytes()
            manifest[name]=dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data))
            output.writestr(name,data)
    result=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifest,
        omitted=omitted,scope='Original geometric logs, failures, frozen inputs and derived views; no new flight or camera capture')
    (package/'evidence_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest),archive_bytes=archive.stat().st_size,
        raw_bytes=sum(x['size'] for x in manifest.values()),omitted=len(omitted))))


if __name__=='__main__':main()

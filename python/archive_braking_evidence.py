"""Archive declared braking flight bytes and the exact analysis/launcher versions."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(manifest_path,analysis_path,historical=None):
    manifest_path=manifest_path.resolve();analysis_path=analysis_path.resolve()
    manifest=read(manifest_path);analysis=read(analysis_path/'summary.json');files={};reports=[]
    for session in manifest['sessions']:
        config_path=Path(session['configuration']).resolve();config=read(config_path)
        if not config_path.is_relative_to(ROOT/'configs') or sha(config_path)!=session['configuration_sha256']:
            raise ValueError('Configuration outside project or changed')
        files[config_path]=session['configuration_sha256']
        conditions=config_path.parent/'conditions.json'
        if conditions.exists():files[conditions]=sha(conditions)
        for name,digest in config['flight_source_sha256'].items():
            if Path(name).name!=name or not name.endswith('.py'):raise ValueError('Unexpected source name')
            files[ROOT/'python'/name]=digest
        if session.get('report_path'):
            path=Path(session['report_path']).resolve()
            if sha(path)!=session['report_sha256']:raise ValueError('Raw report changed')
            reports.append(dict(path=str(path),sha256=sha(path)))
    files[Path(manifest['settings_path']).resolve()]=manifest['settings_sha256']
    if manifest.get('startup_readiness_source_sha256'):
        files[ROOT/'python/braking_startup_readiness.py']=manifest['startup_readiness_source_sha256']
    exact=[(ROOT/'scripts/run-braking-response.ps1',manifest['launcher_sha256']),
           (ROOT/'python/analyze_braking_response.py',analysis['analysis_sha256'])]
    for path,digest in exact:
        if sha(path)!=digest:
            candidate=historical/path.name if historical else None
            if candidate is None or not candidate.exists() or sha(candidate)!=digest:raise ValueError('Exact historical bytes missing: '+path.name)
            path=candidate.resolve()
        files[path]=digest
    for name in ('braking_response.py','braking_response_v1p1.py','trajectory_execution_guard.py','trajectory_timed_reference.py','analyze_trajectory_pilot.py','single_drone_experiment.py'):
        path=ROOT/'python'/name;files.setdefault(path,sha(path))
    for name in ('test_braking_response.py','test_braking_response_v1p1.py','test_braking_response_v1p2.py','test_braking_saved_span.py','test_braking_startup_readiness.py',
                 'test_analyze_braking_response.py','prepare_braking_response.py','prepare_braking_response_v1p1.py',
                 'prepare_braking_response_v1p2.py','prepare_braking_response_v1p3.py','prepare_braking_response_v1p4.py','archive_braking_evidence.py'):
        path=ROOT/'python'/name;files[path]=sha(path)
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('Snapshot source mismatch: '+str(path))
    output=manifest_path.parent/'source_snapshot';output.mkdir(exist_ok=False);rows=[]
    for path,digest in files.items():
        relative=path.relative_to(ROOT);destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if sha(destination)!=digest:raise ValueError('Copy differs')
        rows.append(dict(source=str(path),snapshot=str(destination),sha256=digest))
    versions={}
    for package in ('numpy','scipy','matplotlib','opencv-python','msgpack','msgpack-rpc-python','tornado'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='UNAVAILABLE'
    receipt=dict(status='VERIFIED',manifest=str(manifest_path),manifest_sha256=sha(manifest_path),reports=reports,
        analysis=str(analysis_path/'summary.json'),analysis_sha256=sha(analysis_path/'summary.json'),files=rows,
        runtime_inventory_recorded_after_flight=dict(python=sys.version,packages=versions),
        scope='Declared source/configuration bytes and exact launcher/analyzer archived. Runtime inventory is post-flight; this is not a complete pre-flight transitive dependency lock.')
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True);parser.add_argument('--historical-source-dir',type=Path)
    args=parser.parse_args();archive(args.manifest,args.analysis,args.historical_source_dir)

"""Explicit current-batch evidence archive; no workspace-wide data upload."""
import hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/control_full_flow_20261101'
NETWORK=ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py'

def sha(data):return hashlib.sha256(data).hexdigest()

def package():
    batch=ROOT/'reports/control_full_flow_20261101_v1';campaign=json.loads((batch/'campaign.json').read_text(encoding='utf-8'))
    if campaign['status']!='COMPLETE_WITH_ALL_OUTCOMES' or len(campaign['trials'])!=36:raise ValueError('当前批次不完整')
    folders=[Path(t['run_dir']) for t in campaign['trials']]
    folders += [batch,ROOT/'reports/control_full_flow_matched_20261101_v1',ROOT/'logs/dual_target_airsim_20261031_132033_119671']
    files={p for folder in folders for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pt','.pth')}
    for config in ('curve_live_protocol13.json','curve_live_protocol14.json'):
        d=json.loads((ROOT/'configs'/config).read_text(encoding='utf-8'))
        plans=list(d.get('packages',{}).values())+([d['plans']] if 'plans' in d else [])
        for path in plans:files.add(ROOT/path)
    DEST.mkdir(parents=True,exist_ok=True);archive=DEST/'evidence.zip';entries=[];external=[]
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix();data=path.read_bytes();row=dict(path=name,bytes=len(data),sha256=sha(data))
            if path.name=='abom_net.py':
                if sha(NETWORK.read_bytes())!=row['sha256']:raise ValueError('上游网络副本与固定版本不同')
                external.append(dict(**row,restore_from='tmp/abom_deploy_probe/upstream/abom_net.py'));continue
            z.writestr(name,data);entries.append(row)
    if archive.stat().st_size>90*1024**2:raise ValueError('证据包超过90MiB，请拆分后再上传')
    manifest=dict(schema_version=1,archive='evidence.zip',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,
        files=entries,external_files=external,scope='36 AirSim trials including all 8 timing failures, 192 surrogate jobs, one development request source, startup plans; no UE/Python binaries, credentials or pretrained weights',
        expected=dict(fixed_trials=12,fixed_success=12,online_trials=24,online_success=16,search_jobs=192),
        original_workspace='D:/forAirsim',network_source='https://github.com/xiaofangxd/ABOM/blob/79a2baec700ce0abc6cf7d4cba1e897989d80516/abom_net.py')
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(entries),external_network_copies=len(external),archive_mib=round(archive.stat().st_size/1024**2,2)),ensure_ascii=False))

if __name__=='__main__':package()

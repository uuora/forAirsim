"""Restore byte-exact evidence, refusing to overwrite differing local files."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]

def digest(data):return hashlib.sha256(data).hexdigest()

def target(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('归档路径无效')
    result=(root/Path(*p.parts)).resolve()
    if root not in result.parents:raise ValueError('归档路径越界')
    return result

def restore(root=ROOT,check_only=False):
    root=Path(root).resolve();bundle=root/'experiments/control_full_flow_20261101';manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    archive=bundle/manifest['archive']
    if digest(archive.read_bytes())!=manifest['archive_sha256']:raise ValueError('证据压缩包哈希不符')
    written=0;checked=0
    with zipfile.ZipFile(archive) as z:
        expected={r['path']:r for r in manifest['files']}
        if len(z.namelist())!=len(expected) or set(z.namelist())!=set(expected):raise ValueError('归档清单不符')
        for row in manifest['files']+manifest['external_files']:
            path=target(root,row['path'])
            if path.exists():
                if not path.is_file() or digest(path.read_bytes())!=row['sha256']:raise ValueError('已有文件不同，保留且拒绝覆盖：'+str(path))
                checked+=1;continue
            if check_only:raise FileNotFoundError('缺少证据：'+str(path))
            if row.get('restore_from'):
                source=target(root,row['restore_from'])
                if not source.is_file():raise FileNotFoundError('先运行 scripts/prepare_abom_research.py 获取固定版本上游网络')
                data=source.read_bytes()
            else:data=z.read(row['path'])
            if len(data)!=row['bytes'] or digest(data)!=row['sha256']:raise ValueError('文件哈希不符：'+row['path'])
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(data)
            written+=1
    return dict(status='PASS',written=written,verified_existing=checked,total=len(manifest['files'])+len(manifest['external_files']),expected=manifest['expected'],raw_bytes_rewritten=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(restore(a.root,a.check_only),ensure_ascii=False))

"""Verify or restore the exact byte archive. No AirSim; refuses differing files."""
import argparse,hashlib,json,zipfile,os
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path)
    p.add_argument('--package',type=Path,default=ROOT/'experiments/geometric_control_20261105')
    a=p.parse_args()
    package=a.package.resolve()
    manifest=json.loads((package/'evidence_manifest.json').read_text(encoding='utf-8'))
    archive=package/'evidence.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:raise ValueError('Archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(manifest['files']):raise ValueError('Archive member mismatch')
        verified=[]
        for name,expected in manifest['files'].items():
            parts=PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or ':' in name:raise ValueError('Unsafe member')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Member hash mismatch: '+name)
            if a.destination:
                root=a.destination.resolve()
                if os.name=='nt' and not str(root).startswith('\\\\?\\'):root=Path('\\\\?\\'+str(root))
                dest=root/name
                if dest.is_symlink() or not dest.resolve().is_relative_to(root):raise ValueError('Unsafe destination')
                if dest.exists() and dest.read_bytes()!=data:raise ValueError('Refusing differing file: '+str(dest))
            verified.append((name,data))
        if a.destination:
            for name,data in verified:
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.write_bytes(data)
    print(json.dumps(dict(verified_files=len(verified),restored_to=str(a.destination) if a.destination else None,flight_started=False)))
if __name__=='__main__':main()

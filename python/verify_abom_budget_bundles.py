"""Verify public compressed evidence and replay its accounting audits."""
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile
from audit_abom_budget_research import audit


def main():
    public=Path(__file__).resolve().parents[1]/'experiments/abom_budget_20261029'
    index=json.loads((public/'bundle_index.json').read_text(encoding='utf-8'))
    for name,item in index.items():
        path=public/name
        assert path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with ZipFile(path) as archive:
            assert archive.testzip() is None
            if name=='excluded_diagnostics.zip': continue
            with tempfile.TemporaryDirectory(prefix='abom_evidence_') as temp:
                root=Path(temp).resolve()
                for member in archive.namelist():
                    target=(root/member).resolve()
                    assert target.is_relative_to(root), 'archive path escapes extraction folder'
                archive.extractall(root)
                result=audit(root)
                for filename in ['manifest.json','summary.json','rows.json','paired_comparisons.json','completion.json']:
                    compact=public/name.removesuffix('_raw.zip')/filename
                    if compact.exists(): assert compact.read_bytes()==(root/filename).read_bytes()
                print(name,result['status'])


if __name__=='__main__': main()

"""Read-only re-rendering of audited control records into a new figure folder."""
import argparse,hashlib,json
from pathlib import Path
from audit_airsim_curve_replay_pilot import audit_directory
from run_airsim_curve_replay_pilot import plots


def render(run,output):
    run=Path(run).resolve();run=run/'replay' if (run/'replay/manifest.json').is_file() else run
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    receipt=audit_directory(run,write=False)
    if receipt['evidence_consistency']!='PASS':raise ValueError('evidence audit failed')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((run/'rows.json').read_text(encoding='utf-8'))
    consumed=[run/'manifest.json',run/'rows.json']+[run/r['group_id']/'trace.json' for r in rows]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in consumed}
    output.mkdir(parents=True)
    source_dir=output/'render_source_snapshot';source_dir.mkdir()
    sources=[Path(__file__).resolve(),Path(__file__).with_name('run_airsim_curve_replay_pilot.py').resolve(),
             Path(__file__).with_name('audit_airsim_curve_replay_pilot.py').resolve()]
    source_hashes={str(p):sha(p) for p in sources}
    for p in sources:(source_dir/p.name).write_bytes(p.read_bytes())
    plots(output,rows,manifest['config'],trace_root=run)
    if any(sha(Path(p))!=h for p,h in before.items()):raise RuntimeError('input evidence changed')
    if any(sha(p)!=source_hashes[str(p)] for p in sources):raise RuntimeError('render source changed')
    (output/'render_receipt.json').write_text(json.dumps(dict(status='PASS',input_sha256=before,
        render_source_sha256=source_hashes,audit=receipt,
        figures_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        scope='Plots of recorded state/reference/command data, not new simulation or camera images',ai_assistance=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.run,a.output)

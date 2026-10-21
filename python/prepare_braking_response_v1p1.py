"""New immutable measurement version after the recorded V1 entry failure."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from braking_response_v1p1 import validate_acquisition
ROOT=Path(__file__).resolve().parents[1]


def prepare(output):
    if output.exists(): raise ValueError('New version must not overwrite configurations')
    source=ROOT/'configs/braking_response_v1'; base=json.loads((source/'zero_short.json').read_text(encoding='utf-8'))
    for name,digest in base['flight_source_sha256'].items():
        if hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()!=digest: raise ValueError('V1 source bytes changed: '+name)
    hashes=dict(base['flight_source_sha256'])
    for name in ('braking_response_v1p1.py','braking_response_experiment_v1p1.py'):
        hashes[name]=hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()
    output.mkdir(parents=True); rows=[]
    for name in ('zero_short','low_gate','session_forward','session_reverse'):
        cfg=json.loads((source/(name+'.json')).read_text(encoding='utf-8'))
        cfg['protocol_id']='braking_response_v1p1_20261003'; cfg['flight_source_sha256']=hashes
        cfg['braking_measurement'].update(entry_acquisition_timeout_s=3.,entry_stable_s=.25)
        cfg['evidence_limits'].append('V1.1 acquires actual entry velocity for at most 3 s with a 0.25 s fresh accepted suffix, then retains the 1 s plateau; no weaker entry or stop thresholds')
        validate_acquisition(cfg); path=output/(name+'.json')
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows.append(dict(configuration=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mode=cfg['braking_measurement']['mode'],planned_pulses=len(cfg['braking_measurement']['pulses'])))
    receipt=dict(kind='BRAKING_RESPONSE_CONFIG_FREEZE',status='PREPARED_NOT_EXECUTED',
        prepared_local=datetime.now().astimezone().isoformat(),formal_thesis_effect_results=False,configurations=rows,
        declared_flight_source_sha256=hashes,full_planned_flights=2,full_planned_pulses=16,
        previous_failure='logs/braking_campaign_20261003_221628_9404717/manifest.json',
        changes='Bounded measured entry acquisition before the unchanged hold and stop criteria; no V1 rewrite',
        gates='new zero short, low four-direction gate, separate full sessions audited before the next launch',
        ai_assistance='Implementation and configuration preparation assisted by AI')
    (output/'conditions.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)

"""Retain the one h=1.0 candidate, freeze only the saved-query zero gate repair."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from braking_response_v1p1 import validate_acquisition
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    if output.exists():raise ValueError('New workflow version must be new')
    parent=ROOT/'configs/braking_response_v1p3';receipt=json.loads((parent/'conditions.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True);rows=[]
    for name in ('zero_short','low_gate','session_forward','session_reverse'):
        config=json.loads((parent/(name+'.json')).read_text(encoding='utf-8'))
        for source,digest in config['flight_source_sha256'].items():
            if sha(ROOT/'python'/source)!=digest:raise ValueError('Older declared source changed')
        config['flight_source_sha256']['braking_response_experiment_v1p3.py']=sha(ROOT/'python/braking_response_experiment_v1p3.py')
        config['protocol_id']='braking_response_v1p4_saved_query_gate_20261003'
        config['evidence_limits'].append('Zero hold gate spans saved query receipts, not loop wall time; model h=1.0 remains the one candidate selected before validation')
        validate_acquisition(config);path=output/(name+'.json');path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows.append(dict(configuration=str(path.resolve()),sha256=sha(path),mode=config['braking_measurement']['mode'],planned_pulses=len(config['braking_measurement']['pulses'])))
    receipt.update(status='PREPARED_NOT_EXECUTED',prepared_local=datetime.now().astimezone().isoformat(),configurations=rows,
        parent_freeze=str(parent/'conditions.json'),parent_freeze_sha256=sha(parent/'conditions.json'),
        previous_zero_gate='logs/braking_campaign_20261003_224701_4550403/analysis/summary.json',
        changes='Only zero gate saved-query-span confirmation added; pulse procedure and h=1.0 model unchanged',
        runner='braking_response_experiment_v1p3.py')
    (output/'conditions.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)

"""Retain all prior attempts and each bounded-pipeline attempt with exact inputs."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,expected=None):
    path=Path(p).resolve()
    if not path.is_relative_to(ROOT/'logs') or not path.is_file():raise ValueError('PROJECT_LOG_REQUIRED')
    if expected is not None and sha(path)!=expected:raise ValueError('INPUT_CHANGED')
    return path
def collect(analyses,output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'logs'):raise ValueError('NEW_CAMPAIGN_REQUIRED')
    runs=[];receipts=[]
    for path in analyses:
        path=checked(path);data=json.loads(path.read_text(encoding='utf-8'))
        if data['kind']!='T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS' or data['formal_efficacy_results']:
            raise ValueError('DEVELOPMENT_ANALYSIS_REQUIRED')
        for run in data['runs']:
            for record in run['input_receipts']:checked(record['path'],record['sha256'])
            raw=json.loads((Path(run['run'])/'flight/report.json').read_text(encoding='utf-8'))
            if raw.get('finished_utc') and raw.get('measurement_entered') is False:
                run['incomplete_record_notice']='初始化未通过，未进入主阶段；终态及安全收尾按原独立记录显示'
            runs.append(run)
        receipts.append(dict(path=str(path),sha256=sha(path)))
    if len({r['session_id'] for r in runs})!=len(runs):raise ValueError('DUPLICATED_ATTEMPT')
    missing_terminal=[];nonqualified=[]
    for r in runs:
        raw=json.loads((Path(r['run'])/'flight/report.json').read_text(encoding='utf-8'))
        if not raw.get('finished_utc') or not r['base_evidence']['fresh_ground_closure_verified']:missing_terminal.append(r['session_id'])
        if not r['independent_scale_qualification'].startswith('PASS'):nonqualified.append(r['session_id'])
    result=dict(kind='T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        child_analysis_receipts=receipts,complete_scale_qualified_sessions=[r['session_id'] for r in runs if r['independent_scale_qualification'].startswith('PASS')],
        incomplete_sessions=missing_terminal,nonqualified_sessions=nonqualified,all_attempts=len(runs),
        new_flights_started_by_analysis=0,formal_efficacy_results=False,historical_attempts_retained=True,
        limitations=['All revisions and failures remain visible; current revision qualification is evaluated separately.',
            'All-started attempt denominators include initialization failures; main-frame timing is conditional on entering the main phase.',
            'Unknown terminal endpoints remain unknown; the original interrupted 212-frame ledger remains unscored.',
            'Fixed-height XY development does not establish full 3D optimization or formal method efficacy.'])
    output.mkdir(parents=True);save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save(output/'report.json',result);shutil.copyfile(__file__,output/Path(__file__).name)
    save(output/'artifact_receipt.json',dict(inputs=receipts,files=[dict(path=str(p),sha256=sha(p)) for p in output.iterdir() if p.is_file()]))
    print('retained attempts',len(runs));return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analyses',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.analyses,a.output)

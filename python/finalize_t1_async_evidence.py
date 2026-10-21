"""Bind current documents to the corrected recovery semantics without altering runs."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def finalize(tests):
    out=ROOT/'logs/t1_async_closeout_final_20261005'
    if out.exists():raise ValueError('FINAL_RECEIPT_MUST_BE_NEW')
    timing=ROOT/'logs/t1_async_timing_final_v2_20261005'
    identity=ROOT/'logs/t1_async_identity_analysis_v2_20261005'
    rows=read(identity/'report.json')['runs']
    # An uninitialized observer cannot yield a loss/recovery latency.
    for row in rows[3:]:
        recovery=row['actual_transition_recovery']
        if recovery['loss_receipt_delay_from_actual_hidden_s'] is not None:
            raise ValueError('UNINITIALIZED_LOSS_LATENCY_NOT_CORRECTED')
    if abs(rows[0]['actual_transition_recovery']['loss_receipt_delay_from_actual_hidden_s']-.094)>1e-8:
        raise ValueError('HISTORICAL_VALID_LOSS_CHANGED')
    for folder in (timing,identity,ROOT/'logs/t1_vlx_latency_diagnostic_20261005'):
        for artifact in read(folder/'artifact_receipt.json')['artifacts']:
            if sha(artifact['path'])!=artifact['sha256']:raise ValueError('SEALED_ARTIFACT_CHANGED')
    names=['configs/thesis_simulation_tasks.json','configs/t1_completion_protocol_v1.json',
        'README.md','docs/thesis/第二章_问题建模与仿真平台.md','docs/thesis/第三章_单机感知轨迹优化.md',
        'docs/research-plan-20261004/任务一异步调度与语义时效开发结果.md',
        'docs/research-plan-20261004/硕士论文章节任务与图表安排.md',
        'docs/research-plan-20261004/任务一论文内容与完成标准.md']
    other_before=read(ROOT/names[0])['tasks'][1:]
    out.mkdir();files=[]
    for name in names:
        path=ROOT/name;backup=out/'before'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,backup);before=sha(path)
        text=path.read_text(encoding='utf-8')
        text=text.replace('t1_async_timing_final_20261005','t1_async_timing_final_v2_20261005')
        text=text.replace('t1_async_identity_analysis_20261005','t1_async_identity_analysis_v2_20261005')
        text=text.replace('141项',str(tests)+'项')
        path.write_text(text,encoding='utf-8')
        files.append(dict(path=str(path),before_sha256=before,after_sha256=sha(path)))
    for name in names[:2]:
        path=ROOT/name;value=read(path)
        block=(value['tasks'][0]['async_control_development_20261005'] if 'tasks' in value
               else value['latest_async_control_development'])
        block['tests_passed']=tests
        block['corrected_recovery_semantics']='No loss/recovery latency without a verified pre-hide correct track'
        block['final_closeout_receipt']='logs/t1_async_closeout_final_20261005/receipt.json'
        save(path,value)
        next(row for row in files if row['path']==str(path))['after_sha256']=sha(path)
    if read(ROOT/names[0])['tasks'][1:]!=other_before:raise ValueError('OTHER_TASK_CHANGED')
    docs=ROOT/names[5]
    docs.write_text(docs.read_text(encoding='utf-8')+'\n复算口径更正：没有建立遮挡前正确跟踪的新增试次，其失锁与恢复延迟记为不适用。历史试次有正确跟踪证据的94毫秒失锁记录保留；旧分析目录作为修正前版本保留，当前引用指向修正版。\n',encoding='utf-8')
    next(row for row in files if row['path']==str(docs))['after_sha256']=sha(docs)
    shutil.copyfile(__file__,out/Path(__file__).name)
    source_names=['t1_identity_flight_v2.py','t1_async_control.py','analyze_t1_identity_flight.py',
        'analyze_t1_async_timing.py','profile_t1_vlx_latency.py','prepare_t1_identity_flight_v2.py']
    sources=[]
    for name in source_names:
        source=ROOT/'python'/name;target=out/'source_snapshots'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,target);sources.append(dict(path=str(source),snapshot_path=str(target),sha256=sha(source)))
    save(out/'receipt.json',dict(kind='T1_ASYNC_FINAL_EVIDENCE_BINDING',status='RECORDED_WITH_CORRECTED_METRIC_SCOPE',
        created_utc=datetime.now(timezone.utc).isoformat(),tests_passed=tests,files=files,sources=sources,
        corrected_identity_analysis=str(identity/'report.json'),corrected_timing_analysis=str(timing/'report.json'),
        attempts_four_all_failed=True,attempts_four_all_safely_closed=True,
        formal_complete=False,whole_thesis_complete=False,tasks_two_three_unchanged=True))
    print('FINAL_EVIDENCE_BOUND',len(files),tests)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tests',type=int,required=True)
    finalize(parser.parse_args().tests)

"""Run a real local small LLM; write plans only, never issue flight commands."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from language_plan import SCHEMA, SYSTEM_PROMPT, parse_plan, preview_steps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'downloads' / 'local_llm'

@contextmanager
def local_server(folder):
    executable = ASSETS / 'runtime' / 'llama-server.exe'
    model = ASSETS / 'Qwen3-0.6B-Q8_0.gguf'
    if not executable.is_file() or not model.is_file():
        raise FileNotFoundError('Run python/setup_local_llm.py first')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    with (folder / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([
            str(executable), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
            '-c', '4096', '-np', '1', '-t', '4', '--reasoning', 'off',
        ], stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Local server exited; inspect server.log')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Local server startup timeout')
                time.sleep(0.25)
            yield base
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

def infer(base, instruction):
    payload = dict(model='local-qwen3-0.6b', temperature=0, seed=42, max_tokens=180,
                   messages=[dict(role='system', content=SYSTEM_PROMPT),
                             dict(role='user', content=instruction)],
                   response_format={'type': 'json_schema', 'json_schema': {
                       'name': 'mission_plan', 'strict': True, 'schema': SCHEMA}})
    request = urllib.request.Request(base + '/v1/chat/completions',
                                     data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    seconds = time.perf_counter() - start
    choice = result['choices'][0]
    text = choice['message']['content']
    record = dict(instruction=instruction, raw_output=text, latency_s=seconds,
                  finish_reason=choice['finish_reason'], usage=result.get('usage'),
                  schema_valid=False, plan=None, preview=[], flight_executed=False)
    try:
        if choice['finish_reason'] != 'stop':
            raise ValueError('Incomplete generation')
        plan = parse_plan(text)
        record.update(schema_valid=True, plan=plan, preview=preview_steps(plan))
    except (ValueError, TypeError) as exc:
        record['validation_error'] = str(exc)
    return record

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--instruction')
    group.add_argument('--evaluate', action='store_true')
    parser.add_argument('--split', choices=['development', 'smoke_eval'], default='smoke_eval')
    args = parser.parse_args()
    folder = ROOT / 'logs' / datetime.now().strftime('language_planner_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    report = dict(status='RUNNING', scope='local_CPU_planning_only', model='Qwen3-0.6B-Q8_0',
                  fine_tuned=False, data_review='pending_human_review', runs=[], flight_executed=False,
                  source_sha256={name:hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()
                                 for name in ['language_plan.py', 'run_language_planner.py']})
    manifest = ASSETS / 'manifest.json'
    if manifest.exists():
        report['asset_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
    try:
        if args.evaluate:
            dataset = ROOT / 'datasets' / 'language_tasks_v0' / 'seed.jsonl'
            report['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            report['split'] = args.split
            rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
            rows = [r for r in rows if r['split'] == args.split]
        else:
            rows = [dict(id='interactive', messages=[dict(role='user', content=args.instruction)])]
        with local_server(folder) as base:
            for row in rows:
                instruction = next(m['content'] for m in row['messages'] if m['role'] == 'user')
                record = infer(base, instruction)
                record['id'] = row['id']
                if args.evaluate:
                    expected = json.loads(row['messages'][-1]['content'])
                    record.update(expected=expected, exact_match=record['plan'] == expected)
                report['runs'].append(record)
                (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(record, ensure_ascii=False), flush=True)
        report['status'] = 'COMPLETED'
        report['summary'] = dict(total=len(rows), valid=sum(r['schema_valid'] for r in report['runs']))
        if args.evaluate:
            report['summary']['exact_matches'] = sum(r['exact_match'] for r in report['runs'])
    except BaseException as exc:
        report.update(status='ERROR', error=str(exc))
        raise
    finally:
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(str(folder))

if __name__ == '__main__':
    main()

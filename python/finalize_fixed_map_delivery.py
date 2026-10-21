"""Seal completed evidence; never mutate flight samples or command a vehicle."""
import csv, json, re, shutil, sys, time, unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from fixed_map_mission import OUT, ROOT, read, save, sha, utc, assert_fixed

def main():
    assert_fixed()
    checks=OUT/'final_checks';checks.mkdir(exist_ok=True)
    first=read(OUT/'live_status.json');time.sleep(2);last=read(OUT/'live_status.json')
    assert first['stage']==last['stage']=='READY'
    assert first['run_dir'] is None and last['run_dir'] is None
    assert last['state_timestamp_ns']>first['state_timestamp_ns']
    assert last['landed'] and last['DroneB_landed'] and not last['DroneB_api_control']
    save(checks/'ground_dashboard_updates.json',dict(utc=utc(),status='PASS',
        first={k:v for k,v in first.items() if k not in ('trace','path')},
        last={k:v for k,v in last.items() if k not in ('trace','path')},
        scope='latest persistence revision: startup and live ground updates only; no flight revalidation'))
    for name in ('runtime_geometry.json','preflight.json','active_engine.json','final_ground_proof.json'):
        shutil.copyfile(OUT/name,checks/name)
    test_output=checks/'unit_tests.txt'
    if not test_output.exists():
        suite=unittest.defaultTestLoader.discover(str(ROOT/'python'),pattern='test_fixed_map_mission.py')
        with test_output.open('w',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        assert result.wasSuccessful()
    assert 'Ran 7 tests' in test_output.read_text(encoding='utf-8') and '\nOK' in test_output.read_text(encoding='utf-8')
    for stem in ('plan_and_actual','error_speed_clearance_height'):
        ET.parse(OUT/'analysis'/f'{stem}.svg')
        pdf=(OUT/'analysis'/f'{stem}.pdf').read_bytes()
        assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-1024:]
    for p in (OUT/'analysis').glob('*_origin.csv'):
        with p.open(encoding='utf-8-sig',newline='') as stream:
            rows=csv.reader(stream);header=next(rows);assert len(header)>2
            assert all(len(row)==len(header) for row in rows),str(p)
    doc=(OUT/'DELIVERY.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\((/D:/[^)]+)\)',doc):
        if Path(target[1:])!=OUT/'delivery_receipt.json':assert Path(target[1:]).exists(),target
    immutable=[]
    for folder in ('analysis','final_checks','source_snapshot','source_v3','source_v4','evidence'):
        immutable.extend(p for p in (OUT/folder).rglob('*') if p.is_file())
    for run in OUT.glob('run_*'):
        immutable.extend(p for p in run.rglob('*') if p.is_file())
    immutable += [OUT/name for name in ('DELIVERY.md','operator_provenance.json','drone_envelope_editor.json','ui/160904_READY.png')]
    immutable += [ROOT/p for p in ('python/fixed_map_mission.py','python/fixed_map_dashboard.py',
        'python/analyze_fixed_map_mission.py','python/test_fixed_map_mission.py',
        'python/finalize_fixed_map_delivery.py','scripts/unreal/inspect_flight_envelope.py',
        'docs/fixed-map-mission-interface.md','Run-Fixed-Map-Mission.cmd')]
    save(OUT/'delivery_receipt.json',dict(created_utc=utc(),fixed_version_manifest_sha256=sha(OUT/'fixed_version/manifest.json'),
        primary_evidence=read(OUT/'analysis/primary_evidence_selection.json'),
        current_controller_scope='persistence retry changed after final flight; tested ground startup/updates and safety unit tests only',
        checks=dict(safety_tests_passed=7,csv_rectangular='PASS',svg_xml='PASS',pdf_structure='PASS',report_links='PASS'),
        excluded_mutable='live_status/session_events/camera frames and timestamps; later flights create new evidence',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(immutable))]))
    assert (OUT/'delivery_receipt.json').exists()
    print(json.dumps(dict(receipt=str(OUT/'delivery_receipt.json'),files=len(set(immutable)),tests=7)))

if __name__=='__main__':main()

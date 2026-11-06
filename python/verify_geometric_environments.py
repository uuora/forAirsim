"""Verify every frozen environment, optionally render Qt, with RPC/launch blocked."""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from geometric_validation.environments import CATALOG, ROOT, load_catalog, config_for, task_summary, load_environment_config
from geometric_validation.runtime import load_config, digest, write_json
from geometric_validation.audit import frozen_plan_issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    output=(args.output or ROOT/'reports'/('geometric_environments_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    results=[];images=[]
    with patch('socket.create_connection',side_effect=AssertionError('Verification attempted RPC')), \
         patch('subprocess.Popen',side_effect=AssertionError('Verification attempted process launch')):
        entries=load_catalog()['environments']
        for entry in entries:
            load_environment_config(entry['id'])
            path=config_for(entry['id']);cfg=load_config(path)
            checks={case:frozen_plan_issues(cfg,case) for case in cfg['cases']}
            if any(checks.values()):raise RuntimeError(str(checks))
            results.append(dict(id=entry['id'],label=entry['label'],config=str(path),config_sha256=digest(path),
                task_summary=task_summary(cfg),independent_plan_checks=checks,
                reference_durations_s={k:sum(v['segment_durations']) for k,v in cfg['cases'].items()},
                status='OFFLINE_VALIDATED; no new flight acceptance claimed'))
        if args.render:
            os.environ['QT_QPA_PLATFORM']='offscreen'
            from PyQt5.QtGui import QFont, QFontDatabase
            from PyQt5.QtWidgets import QApplication
            from geometric_validation.qt_panel import GeometricControlWindow
            app=QApplication.instance() or QApplication([])
            if not QFontDatabase().families():
                QFontDatabase.addApplicationFont(str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'))
            app.setFont(QFont('Microsoft YaHei',9))
            window=GeometricControlWindow();window.show()
            for result in results:
                window.environment_combo.setCurrentIndex(window.environment_combo.findData(result['id']))
                for case_index in range(window.case_combo.count()):
                    window.case_combo.setCurrentIndex(case_index)
                    for size in ((1240,880),(800,650)):
                        window.resize(*size);app.processEvents()
                        image=output/(result['id']+'_'+window.case_combo.currentData()+f'_{size[0]}.png')
                        if not window.grab().save(str(image)):raise RuntimeError('Qt rendering failed')
                        if window.active or window.view.snapshot.get('connected'):
                            raise RuntimeError('Preview unexpectedly active or connected')
                        images.append(dict(path=image.name,sha256=digest(image),size=list(size)))
            window.close()
        if 'airsim' in sys.modules:raise RuntimeError('AirSim imported by offline verification')
    sources=list((ROOT/'python/geometric_validation').glob('*.py'))+[Path(__file__).resolve()]
    snapshot=output/'verification_source';snapshot.mkdir()
    for source in sources:
        (snapshot/source.name).write_bytes(source.read_bytes())
    receipt=dict(result='PASS',rpc_attempts=0,flight_started=False,airsim_imported=False,
        catalog_sha256=digest(CATALOG),environments=results,previews=images,
        sources={str(p.relative_to(ROOT)):digest(p) for p in sources},
        note='Qt previews show frozen reference plans, not actual flight or AirSim camera images')
    write_json(output/'verification.json',receipt)
    rows=''.join('<tr><td>'+html.escape(r['label'])+'</td><td>'+f"{r['task_summary']['polyline_length_m']:.1f}"+
        '</td><td>'+f"{r['task_summary']['altitude_min_m']:g}–{r['task_summary']['altitude_max_m']:g}"+
        '</td><td>'+str(r['task_summary']['intermediate_points'])+'</td><td>离线检查通过</td></tr>' for r in results)
    previews=''.join('<h2>'+html.escape(r['label'])+'</h2><img alt="冻结参考的Qt预览" src="'+r['id']+
                     '_bezier_pass_1240.png">' for r in results) if args.render else ''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验环境 · 离线预览</title>
<style>body{font:15px sans-serif;margin:26px;line-height:1.6;color:#252a30}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #d4dce2}img{width:100%;max-width:1240px}h2{font-size:19px}</style>
<h1>实验环境 · 离线预览</h1><p>统一控制器、物理与误差门槛。四组新任务尚未实飞，本页不是飞行通过证据。</p>
<table><tr><th>环境</th><th>航点折线(m)</th><th>航点高度(m)</th><th>中途点</th><th>状态</th></tr>'''+rows+'''</table>
<p>全程三维位置误差≤0.5m；相对同步参考的SO(3)姿态误差≤10°。终点位置≤0.15m、实际速度≤0.15m/s、姿态误差≤5°，保持1s。</p>
<p><a href="verification.json">核验与来源哈希</a></p>'''+previews+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(result='PASS',directory=str(output),environments=len(results),plans=len(results)*3,
                         qt_previews=len(images),rpc=False,flight_started=False),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())

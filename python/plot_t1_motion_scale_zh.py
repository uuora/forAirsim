"""Chinese scientific figures from sealed local motion analysis; no new experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from plot_t1_identity_zh import choose_font, layout_audit, digest, finite, save_json


def replot(analysis, output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('保留已有图表，输出目录必须全新')
    raw=analysis.read_bytes();report=json.loads(raw.decode('utf-8-sig'))
    if report.get('kind')!='T1_MOTION_SCALE_DEVELOPMENT_ANALYSIS':raise ValueError('Motion analysis required')
    font,_=choose_font({ord(c) for c in '中文运动尺度位移指令净0123456789−'},None)
    font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'font.size':10,'pdf.fonttype':42,'axes.unicode_minus':True})
    figures=[];mapping=[]
    for i,run in enumerate(report['runs']):
        response=run.get('command_response',{});states=response.get('sampled_state_rows',[])
        if not states:continue
        mode=run.get('mode');zero=mode=='ZERO_GATE';name=f'试次{i+1}：'+('零输出' if zero else '连续可见运动')
        endpoint=run['motion_scale'];origin=endpoint['start_state']['host_receipt_monotonic_s']
        initial=endpoint['start_state']['position_world_ned_m'][:2]
        t=[r['host_receipt_monotonic_s']-origin for r in states]
        xy=[[1000*(r['position_world_ned_m'][j]-initial[j]) for j in (0,1)] for r in states]
        speed=[1000*(sum(v*v for v in r['velocity_ned_m_s'][:2])**.5) for r in states]
        fig,axes=plt.subplots(2,2,figsize=(13,7.8),layout='constrained');a,b,c,d=axes.flat
        a.plot(t,[p[0] for p in xy],label='X位移',color='#3977a6');a.plot(t,[p[1] for p in xy],label='Y位移',color='#b87835')
        a.scatter([t[-1]],[(xy[-1][0]**2+xy[-1][1]**2)**.5],s=35,color='#246346',label='端点净位移')
        a.set_title(f'端点净位移 {endpoint["net_xy_displacement_m"]*1000:.3f} 毫米')
        a.set_ylabel('相对主段起点位移（毫米）');a.set_xlabel('主段起点后时间（秒）');a.legend(fontsize=9)
        b.plot(t,speed,color='#246346',label='实际XY速度（采样）')
        commands=response.get('rows',[])
        if commands:
            ct=[r['actual_dispatch_s']-origin for r in commands]
            b.scatter(ct,[1000*r['actual_command_speed_m_s'] for r in commands],s=18,label='实际派发目标速度',color='#73559a')
            desired=[r for r in commands if finite(r.get('desired_speed_m_s'))]
            b.scatter([r['actual_dispatch_s']-origin for r in desired],[1000*r['desired_speed_m_s'] for r in desired],
                      marker='x',s=23,label='规划期望速度',color='#ba852c')
        b.set_title('实际响应与目标指令');b.set_ylabel('速度（毫米/秒）');b.set_xlabel('主段起点后时间（秒）');b.legend(fontsize=8)
        rows=run['continuous_visibility']['rows'];vt=[r['readback_s']-origin for r in rows]
        c.plot(vt,[r['target_mask_center_error_norm'] for r in rows],'.-',color='#3977a6',label='目标掩膜中心偏差')
        c.set_title('图像中心偏差（描述性诊断）');c.set_ylabel('归一化中心距离');c.set_xlabel('实际图像回读时间（秒）');c.legend(fontsize=9)
        d.plot(vt,[r['target_visible_pixels'] for r in rows],'.-',color='#246346',label='指定目标可见像素')
        d.axhline(500,ls='--',color='#b64240',label='预声明门槛：500像素')
        d.set_title(f'主段保存 {len(rows)} / {run["measured_contract"]["expected_slots"]} 槽')
        d.set_ylabel('可见像素数');d.set_xlabel('实际图像回读时间（秒）');d.legend(fontsize=9)
        for ax in axes.flat:ax.grid(alpha=.2)
        conclusion={'PASS_ENDPOINT_MOTION_SCALE':'达到20毫米尺度','FAIL_ENDPOINT_MOTION_SCALE_NOT_REACHED':'未达到20毫米尺度',
                    'PASS_ZERO_GATE_DRIFT':'零输出漂移通过'}.get(run['motion_scale_qualification'],'未满足全部独立判据（见数据表）')
        fig.suptitle(name+'；'+conclusion+'\n开发验证；端点净位移与采样路程分别报告，正式轨迹效果尚未验证',fontsize=13)
        key=f'motion_scale_{i+1}_zh';figures.append((key,fig))
        mapping.append(dict(figure=key,session_id=run['session_id'],qualification=run['motion_scale_qualification'],
            endpoint=endpoint,actual_state_points=[dict(time_s=x,displacement_xy_mm=y,speed_xy_mm_s=z) for x,y,z in zip(t,xy,speed)],
            command_points=commands,actual_mask_points=rows,window_diagnostics=run.get('center_windows')))
    if not figures:raise ValueError('No saved motion state points to plot')
    for _,fig in figures:fig.canvas.draw()
    strings=[text.get_text() for _,fig in figures for text in fig.findobj(Text) if text.get_visible()]
    _,font_receipt=choose_font({ord(c) for s in strings for c in s if not c.isspace()},font)
    layouts=[layout_audit(fig) for _,fig in figures]
    if any(row['text_bounds_status']!='PASS' for row in layouts):raise ValueError('图表布局检查失败')
    output.mkdir(parents=True,exist_ok=False);artifacts=[]
    for key,fig in figures:
        for suffix in ('png','pdf'):
            p=output/(key+'.'+suffix);fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');artifacts.append(p)
        plt.close(fig)
    data=output/'plot_data_mapping.json'
    save_json(data,dict(kind='T1_CHINESE_MOTION_SCALE_POINT_MAPPING',figures=mapping,
        attempts_without_motion_data=[r['session_id'] for r in report['runs'] if not r.get('command_response',{}).get('sampled_state_rows')],
        formal_efficacy_results=False,new_experiment=False));artifacts.append(data)
    sources=[]
    for name in ('plot_t1_motion_scale_zh.py','plot_t1_identity_zh.py'):
        p=Path(__file__).with_name(name).resolve();s=output/name;shutil.copyfile(p,s)
        sources.append(dict(path=str(p),snapshot_path=str(s),sha256=digest(p)));artifacts.append(s)
    if analysis.read_bytes()!=raw:raise ValueError('Input changed during plotting')
    receipt=dict(kind='T1_CHINESE_MOTION_SCALE_FIGURE_RECEIPT',created_utc=datetime.now(timezone.utc).isoformat(),
        input_analysis=dict(path=str(analysis),sha256=digest(analysis)),font=font_receipt,sources=sources,
        layout_checks=layouts,visual_review=dict(status='PENDING'),formal_efficacy_results=False,new_experiment=False,
        artifacts=[dict(path=str(p),sha256=digest(p)) for p in artifacts])
    save_json(output/'artifact_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=replot(args.analysis,args.output);print(json.dumps(dict(figures=len(result['layout_checks']),font_coverage=result['font']['glyph_coverage'])))

"""Chinese static figures from sealed timing/own-state analysis, without new flights."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

from plot_t1_identity_zh import choose_font, layout_audit, digest, save_json


def plot(analysis, output):
    analysis=Path(analysis).resolve(); output=Path(output).resolve()
    if output.exists(): raise FileExistsError('图表目录必须全新，保留原图')
    raw=analysis.read_bytes(); data=json.loads(raw.decode('utf-8-sig'))
    if data['kind']!='T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS': raise ValueError('计时与停稳分析报告格式不符')
    font,_=choose_font({ord(c) for c in '任务一中文计时停稳速度毫米秒'},None)
    font_manager.fontManager.addfont(str(font)); prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'font.size':10,'axes.unicode_minus':True,'pdf.fonttype':42})
    figures=[]; mapping=[]
    for n,r in enumerate(data['runs']):
        states=r.get('sampled_state_rows',[])
        if not states: continue
        endpoint=r['endpoint']; origin=endpoint['start_state']['host_receipt_monotonic_s']; start=endpoint['start_state']['position_world_ned_m']
        points=[dict(time_s=s['host_receipt_monotonic_s']-origin,
            net_xy_mm=math.hypot(*(s['position_world_ned_m'][i]-start[i] for i in range(2)))*1000,
            speed_xy_mm_s=math.hypot(*s['velocity_ned_m_s'][:2])*1000,
            state_timestamp_ns=s.get('state_timestamp_ns')) for s in states]
        timing=[dict(frame_id=f['frame_id'],slot_index=f['slot_index'],
            frame_processing_s=f['recorded_timing']['frame_processing']['after_due_s'],
            gate_s=f['recorded_timing']['gate']['after_due_s'],
            historical_full_s=f['recorded_timing']['historical_full']['after_due_s']) for f in r['frame_timing_rows']]
        commands=r['base_evidence']['target_command_rows']
        command_points=[dict(command_id=c.get('command_id'),time_s=c['dispatch_monotonic_s']-origin,
            command_y_mm_s=c['velocity_xy_m_s'][1]*1000,duration_s=c['duration_s']) for c in commands
            if c['dispatch_monotonic_s']>=origin]
        strict='V7' in r['architecture']; threshold=.5 if strict else 5.
        duration=r.get('plan_check',{}).get('expected_main_duration_s',len(timing)*.4)
        label=('零输出' if r['mode']=='ZERO_GATE' else f'{duration:g}秒运动')
        version=next((v for v in ('V7P3','V7P2','V7P1') if v in r['architecture']), 'V7' if strict else 'V6P1')
        name=f'任务一：{label}与停稳计时（{version}，试次{n+1}）'
        fig,axes=plt.subplots(2,2,figsize=(12.8,8.2)); fig.suptitle(name,fontsize=15,y=.975)
        ax=axes[0,0]
        ax.plot([p['time_s'] for p in points],[p['net_xy_mm'] for p in points],color='#3379a5',linewidth=1.8)
        if r['mode']=='SHORT_XY': ax.axhline(20.,color='#aa7440',linestyle='--',label='预先声明的20毫米尺度')
        ax.set(title='自身状态测得的XY位移',xlabel='相对端点起始时刻（秒）',ylabel='相对起点距离（毫米）')
        if r['mode']=='SHORT_XY': ax.legend(fontsize=9,loc='best')
        ax=axes[0,1]
        ax.plot([p['time_s'] for p in points],[p['speed_xy_mm_s'] for p in points],color='#497b54',linewidth=1.2)
        ax.axhline(threshold,color='#aa7440',linestyle='--',label=f'本版本停稳速度阈值：{threshold:g}毫米/秒')
        ax.set(title='实际速度与停稳阈值',xlabel='相对端点起始时刻（秒）',ylabel='XY速度（毫米/秒）'); ax.legend(fontsize=9)
        ax=axes[1,0]
        for key,label,color in (('gate_s','控制决策','#397ba6'),('frame_processing_s','帧处理完成','#527d53'),
                ('historical_full_s','完整完成（含停稳）','#b16845')):
            ax.plot([x['slot_index'] for x in timing],[float('nan') if x[key] is None else x[key] for x in timing],
                label=label,color=color,linewidth=1.3,marker='.',markersize=4)
        ax.axhline(.4,color='#777777',linestyle='--',label='原0.4秒帧截止点')
        ax.set(title='各阶段实际耗时',xlabel='主测量帧序号',ylabel='从本帧计划起点计时（秒）'); ax.legend(fontsize=8.5)
        ax=axes[1,1]
        if command_points:
            ax.vlines([c['time_s'] for c in command_points],0,[c['command_y_mm_s'] for c in command_points],color='#397ba6',linewidth=1)
            ax.scatter([c['time_s'] for c in command_points],[c['command_y_mm_s'] for c in command_points],s=16,color='#397ba6',label='实际非零目标指令')
            ax.legend(fontsize=9)
        else: ax.text(.5,.5,'本试次没有非零目标指令',ha='center',va='center',transform=ax.transAxes)
        ax.set(title='目标指令的实际派发',xlabel='相对端点起始时刻（秒）',ylabel='Y指令速度（毫米/秒）')
        for ax in axes.flat: ax.grid(alpha=.18)
        fig.text(.04,.022,'受控开发数据；完整完成的原截止点保留。严格停稳是采样判据，正式四方法对照尚未完成。',fontsize=9,color='#565e67')
        fig.tight_layout(rect=(0,.055,1,.95)); fig.canvas.draw()
        for ax in axes.flat:
            xl,xh=ax.get_xlim(); yl,yh=ax.get_ylim()
            ax.set_xticks([x for x in ax.get_xticks() if xl<=x<=xh]); ax.set_yticks([y for y in ax.get_yticks() if yl<=y<=yh])
        fig.canvas.draw()
        key='settlement_'+str(n+1)+'_zh'; figures.append((key,fig))
        mapping.append(dict(figure=key,session_id=r['session_id'],own_state_points=points,
            timing_points=timing,target_command_points=command_points,stationary_threshold_mm_s=threshold,
            endpoint_net_xy_m=endpoint['net_xy_displacement_m'],endpoint_scale_status=r['endpoint_scale_status'],
            raw_source_receipts=r['input_receipts']))
    strings=[t.get_text() for _,fig in figures for t in fig.findobj(Text) if t.get_visible()]
    _,font_receipt=choose_font({ord(c) for s in strings for c in s if not c.isspace()},font)
    layouts=[layout_audit(fig) for _,fig in figures]
    if not figures or any(x['text_bounds_status']!='PASS' for x in layouts): raise ValueError('中文图表布局核查未通过')
    output.mkdir(parents=True); files=[]
    for key,fig in figures:
        for suffix in ('png','pdf'):
            p=output/(key+'.'+suffix); fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white'); files.append(p)
        plt.close(fig)
    p=output/'plot_data_mapping.json'; save_json(p,dict(kind='T1_CHINESE_SETTLEMENT_POINT_MAPPING',figures=mapping)); files.append(p)
    for name in ('plot_t1_settlement_zh.py','plot_t1_identity_zh.py'):
        source=Path(__file__).with_name(name); p=output/name; shutil.copyfile(source,p); files.append(p)
    if analysis.read_bytes()!=raw: raise ValueError('图表输入发生变化')
    receipt=dict(kind='T1_CHINESE_SETTLEMENT_FIGURE_RECEIPT',created_utc=datetime.now(timezone.utc).isoformat(),
        input_analysis=dict(path=str(analysis),sha256=digest(analysis)),font=font_receipt,layout_checks=layouts,
        visual_review=dict(status='PENDING'),artifacts=[dict(path=str(p),sha256=digest(p)) for p in files],
        new_flights=0,formal_efficacy_results=False)
    save_json(output/'artifact_receipt.json',receipt)
    print('中文图表',len(figures),'字体',font_receipt['glyph_coverage'],'布局全部通过')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); plot(args.analysis,args.output)

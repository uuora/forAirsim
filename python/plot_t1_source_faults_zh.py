"""Chinese scientific figures for independently scored source perturbations."""
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

from plot_t1_identity_zh import choose_font, layout_audit, digest, save_json

LABELS={'baseline':'零输出门控','delivery_delay':'真实回复延迟交付','occlusion':'遮挡与返回','glint':'小白斑干扰'}


def plot(analysis, output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_CHINESE_SOURCE_FAULT_FIGURE_FOLDER_REQUIRED')
    raw=analysis.read_bytes();data=json.loads(raw.decode('utf-8-sig'))
    font,_=choose_font({ord(c) for c in '任务一中文源证据扰动'},None)
    font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'font.size':10,'axes.unicode_minus':True,'pdf.fonttype':42})
    figures=[];maps=[]
    for i,r in enumerate(data['runs'],1):
        base=r['base_evidence'];epoch=base['measured_epoch_monotonic_s'];condition=r['source_perturbation']['condition']
        fig,axes=plt.subplots(2,2,figsize=(13.4,8.5))
        fig.suptitle(f'任务一：{LABELS[condition]}（试次{i}，受控开发）',fontsize=15,y=.975)
        frame_points=r['source_fault_frame_rows'];ax=axes[0,0]
        for status,label,height,color in [('CORRECT','正确目标',1.,'#467c53'),('NO_OUTPUT','无目标输出',0.,'#7c858d')]:
            points=[f for f in frame_points if f['identity']==status]
            if points:ax.scatter([f['relative_readback_s'] for f in points],[height]*len(points),s=13,c=color,label=label)
        wrong=[f for f in frame_points if f['identity'] not in ('CORRECT','NO_OUTPUT')]
        if wrong:ax.scatter([f['relative_readback_s'] for f in wrong],[-1.]*len(wrong),s=23,c='#b34e46',marker='x',label='错选或无法匹配')
        if condition=='occlusion':ax.axvspan(20.,24.,alpha=.12,color='#827491',label='计划遮挡阶段')
        if condition=='glint':ax.axvspan(20.,40.,alpha=.12,color='#c58a38',label='计划白斑阶段')
        for j,h in enumerate(r['reply_delivery_audit']['holds']):
            if h['hold_started_monotonic_s'] is not None and h['hold_completed_monotonic_s'] is not None:
                ax.axvspan(h['hold_started_monotonic_s']-epoch,h['hold_completed_monotonic_s']-epoch,
                    alpha=.12,color='#c58a38',label='实际后台交付等待' if j==0 else None)
        ax.set(title='保存主帧的独立身份评分',xlabel='相对主阶段开始（秒）',ylabel='身份输出状态',yticks=[-1,0,1],
            yticklabels=['错选／无法匹配','无输出','正确'],ylim=(-1.4,1.6));ax.legend(fontsize=8,loc='lower right')
        start=r['endpoint']['start_state'];origin=start['position_world_ned_m'];points=[]
        for s in r['sampled_state_rows']:
            net=((s['position_world_ned_m'][0]-origin[0])**2+(s['position_world_ned_m'][1]-origin[1])**2)**.5*1000
            points.append(dict(relative_time_s=s['host_receipt_monotonic_s']-epoch,net_xy_mm=net,state_timestamp_ns=s['state_timestamp_ns']))
        ax=axes[0,1];ax.plot([p['relative_time_s'] for p in points],[p['net_xy_mm'] for p in points],c='#397ba6',linewidth=1.6)
        if r['mode']=='SHORT_XY':ax.axhline(20.,linestyle='--',c='#ad7740',label='保留的20毫米尺度');ax.legend(fontsize=9)
        ax.set(title='自身状态测得的XY位移',xlabel='相对主阶段开始（秒）',ylabel='相对起点距离（毫米）')
        ax=axes[1,0];time_points=[]
        for key,label,color in [('frame_processing','帧处理完成','#527d53'),('historical_full','完整完成（含停稳）','#b16845')]:
            xs=[];ys=[]
            for f in r['frame_timing_rows']:
                xs.append(f['slot_index']);v=f['recorded_timing'][key]['after_due_s'];ys.append(float('nan') if v is None else v)
            ax.plot(xs,ys,label=label,c=color,linewidth=1.2,marker='.',markersize=3)
        ax.axhline(.4,linestyle='--',c='#777777',label='原0.4秒帧截止点')
        ax.set(title='原完成时钟与截止点',xlabel='主测量帧序号',ylabel='从本帧计划起点计时（秒）');ax.legend(fontsize=8,loc='upper right')
        ax=axes[1,1];requests=[]
        for record in base['sanitized_transport_trace']:
            a=record.get('backend_started_monotonic_s');b=record.get('backend_completed_monotonic_s')
            if a is None or b is None or record['request_binding']['readback_receipt_monotonic_s']<epoch:continue
            delivery=record.get('controlled_reply_delivery',{})
            hold=0.
            if delivery.get('hold_completed_monotonic_s') is not None:
                hold=delivery['hold_completed_monotonic_s']-delivery['hold_started_monotonic_s']
            requests.append(dict(request_id=record['request_binding']['request_id'],real_service_wall_s=b-a,
                actual_controlled_hold_s=hold,source_readback_s=record['request_binding']['readback_receipt_monotonic_s']))
        x=list(range(1,len(requests)+1));service=[q['real_service_wall_s'] for q in requests];holds=[q['actual_controlled_hold_s'] for q in requests]
        if requests:
            ax.bar(x,service,color='#397ba6',label='实际后端耗时')
            if any(holds):ax.bar(x,holds,bottom=service,color='#b88944',label='实际人工交付等待')
            ax.legend(fontsize=8)
        else:ax.text(.5,.5,'没有主段后端完成回执',ha='center',va='center',transform=ax.transAxes)
        ax.set(title='服务耗时与人工等待分别记录',xlabel='主段请求完成序号',ylabel='耗时（秒）')
        for ax in axes.flat:ax.grid(alpha=.16)
        fig.text(.035,.020,'保存帧是相关采样；人工等待不计为自然服务延迟。运动尺度与扰动行为分别评价，正式四方法对照仍待开展。',fontsize=9,color='#59616b')
        fig.tight_layout(rect=(0,.055,1,.95));fig.canvas.draw()
        for ax in axes.flat:
            lo,hi=ax.get_xlim();ax.set_xticks([v for v in ax.get_xticks() if lo<=v<=hi])
            if ax is not axes[0,0]:
                lo,hi=ax.get_ylim();ax.set_yticks([v for v in ax.get_yticks() if lo<=v<=hi])
        fig.canvas.draw();key=f'source_fault_{i}_zh';figures.append((key,fig))
        maps.append(dict(figure=key,session_id=r['session_id'],condition=condition,frame_points=frame_points,
            own_state_points=points,timing_points=r['frame_timing_rows'],request_points=requests,
            source_receipts=r['input_receipts'],fault_qualification=r['independent_fault_qualification'],scale_status=r['endpoint_scale_status']))
    strings=[t.get_text() for _,fig in figures for t in fig.findobj(Text) if t.get_visible()]
    _,font_receipt=choose_font({ord(c) for s in strings for c in s if not c.isspace()},font)
    layouts=[layout_audit(fig) for _,fig in figures]
    if not figures or any(x['text_bounds_status']!='PASS' for x in layouts):raise ValueError('CHINESE_FAULT_FIGURE_LAYOUT_FAILED')
    output.mkdir(parents=True);files=[]
    for key,fig in figures:
        for suffix in ('png','pdf'):
            p=output/(key+'.'+suffix);fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');files.append(p)
        plt.close(fig)
    p=output/'plot_data_mapping.json';save_json(p,dict(kind='T1_CHINESE_SOURCE_FAULT_MAPPING',figures=maps));files.append(p)
    for name in ('plot_t1_source_faults_zh.py','plot_t1_identity_zh.py'):
        p=output/name;shutil.copyfile(Path(__file__).with_name(name),p);files.append(p)
    if analysis.read_bytes()!=raw:raise ValueError('SOURCE_FAULT_FIGURE_INPUT_CHANGED')
    save_json(output/'artifact_receipt.json',dict(kind='T1_CHINESE_SOURCE_FAULT_FIGURE_RECEIPT',created_utc=datetime.now(timezone.utc).isoformat(),
        input_analysis=dict(path=str(analysis),sha256=digest(analysis)),font=font_receipt,layout_checks=layouts,
        visual_review=dict(status='PENDING'),artifacts=[dict(path=str(p),sha256=digest(p)) for p in files],new_flights=0,formal_efficacy_results=False))
    print('中文源证据扰动图表',len(figures),'字体',font_receipt['glyph_coverage'],'布局通过')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.analysis,a.output)

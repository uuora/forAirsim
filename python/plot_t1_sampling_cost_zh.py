"""Chinese source-bound checkpoint costs and missing-slot context."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

from plot_t1_identity_zh import choose_font,layout_audit,digest,save_json


def plot(analysis,output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_COST_FIGURE_FOLDER_REQUIRED')
    report=json.loads(analysis.read_text(encoding='utf-8-sig'))
    if report['status']!='PASS_OFFLINE_CHECKPOINT_ACCOUNTING' or report['new_flights']!=0:
        raise ValueError('OFFLINE_CHECKPOINT_ANALYSIS_REQUIRED')
    font,_=choose_font({ord(c) for c in '中文采样成本'},None)
    font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'font.size':10,'axes.unicode_minus':True,'pdf.fonttype':42})
    runs=report['runs'];pilot=next(r for r in runs if r['session_id']=='t1_scale_50_v9_20261006')
    repeat=next(r for r in runs if r['session_id']=='t1_scale_50_v9_rep3_20261007')
    import csv
    csv_path=analysis.parent/'frame_checkpoint_costs.csv'
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:frames=list(csv.DictReader(stream))
    rows=[f for f in frames if f['session_id']==repeat['session_id']]
    if len(rows)!=repeat['measured_main_frames']:raise ValueError('FIGURE_FRAME_COUNT_MISMATCH')
    fig,axes=plt.subplots(2,2,figsize=(14.8,9.2))
    fig.suptitle('任务一：串行采样成本与缺槽上下文（既有开发记录）',fontsize=16,y=.98)
    stages=list(repeat['stage_statistics']);y=np.arange(len(stages));ax=axes[0,0]
    for shift,run,color,label in [(-.18,pilot,'#397ba6','50毫米先导：400帧'),(.18,repeat,'#ac6847','50毫米重复3：393帧')]:
        values=[run['stage_total_seconds'][k]/run['measured_main_frames']*1000 for k in stages]
        ax.barh(y+shift,values,height=.34,color=color,label=label)
    short=['采集（RPC与等待）','采集后至RGB保存开始','RGB保存及校验','RGB后至观察器开始','观察器（含检测与关联）','观察器后至表面结束','表面后至控制提交','控制提交至决策可用','决策后至媒体处理结束']
    ax.set(yticks=y,yticklabels=short,xlabel='逐阶段会话内平均耗时（毫秒）',title='九个主线程连续检查点区间');ax.invert_yaxis();ax.legend(fontsize=8)
    ax=axes[0,1];x=np.arange(len(runs))
    for shift,key,color,label in [(-.18,'median_ms','#158775','中位数'),(.18,'p95_nearest_rank_ms','#397ba6','95分位数（最近秩）')]:
        values=[r['producer_checkpoint_span'][key] for r in runs]
        bars=ax.bar(x+shift,values,width=.34,color=color,label=label)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+7,f'{value:.0f}',ha='center',fontsize=9)
    ax.axhline(400,c='#b56d43',ls='--',label='400毫秒采样周期')
    ax.set(xticks=x,xticklabels=['零输出门控\n20帧','50毫米先导\n400帧','50毫米重复3\n393帧'],ylabel='从采集开始至帧处理结束（毫秒）',title='处理跨度分布：停止尾段另计',ylim=(0,520));ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,0]
    slots=[int(r['slot_index']) for r in rows]
    ax.plot(slots,[float(r['start_offset_s'])*1000 for r in rows],c='#158775',lw=1,label='采集开始偏移（相对本槽计划）')
    ax.plot(slots,[float(r['producer_checkpoint_span_s'])*1000 for r in rows],c='#397ba6',lw=1,label='本帧主线程处理跨度')
    ax.axhline(400,c='#b56d43',ls='--',label='400毫秒周期')
    for gap in report['missing_slot_context']:ax.axvline(gap['missing_slot'],c='#bf554c',lw=.6,alpha=.6)
    ax.set(xlabel='重复3计划主槽序号',ylabel='耗时或计划偏移（毫秒）',title='红色竖线标记7个未采集槽');ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,1];gaps=report['missing_slot_context'];x=np.arange(len(gaps));bottom=np.zeros(len(gaps))
    for key,color,label in [('previous_capture_start_offset_s','#158775','上一帧采集开始偏移'),('previous_producer_span_s','#397ba6','上一帧主线程处理'),('processing_end_to_slot_check_unattributed_s','#b47b44','处理结束至缺槽检查（未细分）')]:
        vals=np.array([r[key]*1000 for r in gaps]);ax.bar(x,vals,bottom=bottom,color=color,width=.68,label=label);bottom+=vals
    if not all(g['missing_slot']==g['previous_slot']+1 for g in gaps):raise ValueError('NONADJACENT_GAP_CONTEXT')
    ax.axhline(800,c='#bf554c',ls='--',label='下一槽截止（相对上一槽计划：800毫秒）')
    ax.set(xticks=x,xticklabels=[str(g['missing_slot']) for g in gaps],xlabel='未采集主槽序号',ylabel='从上一槽计划至缺槽检查（毫秒）',title='缺槽检查时刻：相邻检查点合计',ylim=(0,1250));ax.legend(fontsize=8,loc='upper left')
    for ax in axes.flat:ax.grid(alpha=.15,axis='x' if ax is axes[0,0] else 'y');ax.set_axisbelow(True)
    fig.text(.025,.027,'帧为会话内记录，停止分支可与后帧重叠，单列且未加入串行成本；已有时钟不能细分YOLO或事件刷新，不能据此证明单一原因。',fontsize=9,color='#59616b')
    fig.tight_layout(rect=(0,.055,1,.95));fig.canvas.draw()
    required={ord(c) for t in fig.findobj(Text) for c in t.get_text() if not c.isspace()}
    _,font_receipt=choose_font(required,font);layout=layout_audit(fig)
    if layout['text_bounds_status']!='PASS':raise ValueError('SAMPLING_COST_LAYOUT_FAILED')
    output.mkdir(parents=True);artifacts=[]
    for suffix in ('png','pdf'):
        p=output/('sampling_cost_zh.'+suffix);fig.savefig(p,dpi=180);artifacts.append(dict(path=str(p),sha256=digest(p)))
    for source in (Path(__file__),Path(__file__).with_name('plot_t1_identity_zh.py')):
        p=output/source.name;p.write_bytes(source.read_bytes());artifacts.append(dict(path=str(p),sha256=digest(p)))
    save_json(output/'artifact_receipt.json',dict(kind='T1_CHINESE_SAMPLING_COST_FIGURE_RECEIPT',created_utc=datetime.now(timezone.utc).isoformat(),
        input_analysis=dict(path=str(analysis),sha256=digest(analysis)),input_csv=dict(path=str(csv_path),sha256=digest(csv_path)),
        font=font_receipt,layout_checks=[layout],artifacts=artifacts,visual_review=dict(status='PENDING'),
        new_flights=0,new_model_calls=0,formal_efficacy_results=False))
    plt.close(fig);print('Chinese sampling cost figure, glyph and layout PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.analysis,a.output)

"""Export a read-only, source-bound research overview from saved T1 evidence.

No simulation, model evaluation or network calls. A new output folder is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from plot_t1_identity_zh import choose_font

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'detector': ROOT / 'logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json',
    'campaign': ROOT / 'logs/t1_process_campaign_with_supplement_v11_20261008/report.json',
    'age': ROOT / 'logs/t1_process_source_age_figures_zh_v11_v1p3_20261008/data.json',
}
BLUE, TEAL, ORANGE, GRAY = '#3478A3', '#228579', '#D28A3B', '#88939B'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def polish(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#B7C2C9')
    ax.tick_params(color='#B7C2C9', labelcolor='#33434E')
    ax.grid(axis='y', color='#DCE4E8', linewidth=.6, alpha=.65)
    ax.set_axisbelow(True)


def finish(fig, output, name, caption, records):
    fig.subplots_adjust(bottom=.24, top=.82, left=.13 if name.startswith('03_') else .085,
                        right=.97, wspace=.33)
    fig.text(.085, .07, caption, va='bottom', fontsize=9, color='#596B77', linespacing=1.7)
    # Freeze only ticks inside the view; auto locators also create hidden outer ticks.
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lower, upper = sorted(limits)
            ticks = axis.get_majorticklocs()
            axis.set_ticks([t for t in ticks if lower <= t <= upper])
    # Select a font against all actual plot strings, including captions and ticks.
    fig.canvas.draw()
    chars = {ord(c) for t in fig.findobj(Text) for c in t.get_text() if not c.isspace()}
    font, _ = choose_font(chars, None)
    prop = font_manager.FontProperties(fname=str(font))
    for text in fig.findobj(Text):
        size = text.get_fontsize()
        text.set_fontproperties(prop)
        text.set_fontsize(size)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width + 2 or box.y1 > fig.bbox.height + 2:
            outside.append(text.get_text())
    if outside:
        raise ValueError('TEXT_OUTSIDE_CANVAS: ' + repr(outside))
    for suffix in ('png', 'pdf', 'svg'):
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white')
        records.append({'path': str(path), 'sha256': sha(path)})
    plt.close(fig)
    return {'font_path': str(font), 'font_sha256': sha(font), 'outside_text': outside}


def detector_plot(d, output, records):
    models = ['original_COCO_sports_ball', 'specialist_balloon']
    episodes = sorted(d['summary_by_episode_and_model'][models[0]])
    session_rows, frame_rows = [], []
    for model in models:
        for ep in episodes:
            frames = [r for r in d['frames'] if r['model'] == model and r['episode_id'] == ep]
            assert len(frames) == 24 and all(r['status'] == 'RETURNED' for r in frames)
            tp = sum(r['score']['true_positive'] for r in frames)
            fp = sum(r['score']['false_positive'] for r in frames)
            fn = sum(r['score']['false_negative'] for r in frames)
            saved = d['summary_by_episode_and_model'][model][ep]
            assert (tp, fp, fn) == (saved['true_positive'], saved['false_positive'], saved['false_negative'])
            session_rows.append(dict(model=model, episode_id=ep, frames=len(frames), tp=tp, fp=fp,
                                     fn=fn, recall=tp/(tp+fn), precision=tp/(tp+fp),
                                     median_latency_ms=float(np.median([r['elapsed_ns']/1e6 for r in frames]))))
            for r in frames:
                frame_rows.append(dict(model=model, episode_id=ep, frame_id=r['frame_id'],
                                       latency_ms=r['elapsed_ns']/1e6, image_sha256=r['image_sha256'],
                                       tp=r['score']['true_positive'], fp=r['score']['false_positive'],
                                       fn=r['score']['false_negative']))
    save_csv(output/'detector_sessions.csv', session_rows)
    save_csv(output/'detector_frames.csv', frame_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.3))
    for ax, metric, title in zip(axes[:2], ['recall', 'precision'], ['实例召回率', '检测精确率']):
        for j, ep in enumerate(episodes):
            vals = [next(r[metric]*100 for r in session_rows if r['model']==m and r['episode_id']==ep) for m in models]
            color, marker = [(BLUE, 'o'), (TEAL, 'D')][j]
            offset = [-.025, .025][j]
            ax.plot(np.array([0, 1])+offset, vals, marker=marker, color=color, lw=1.2,
                    markersize=7, label=f'测试会话 {j+1}')
        ax.set_xticks([0, 1], ['原 COCO 映射', '专用气球模型'])
        ax.set(xlim=(-.25, 1.25), ylim=(0, 108), ylabel=title+'（%）', title=title+'：相同会话配对')
        ax.legend(frameon=False, fontsize=9, loc='lower right')
        polish(ax)
    for model, color in zip(models, [GRAY, TEAL]):
        vals = np.sort([r['latency_ms'] for r in frame_rows if r['model']==model])
        axes[2].step(vals, np.arange(1, len(vals)+1)/len(vals), where='post', color=color,
                     lw=1.4, label='原 COCO 映射' if model==models[0] else '专用气球模型')
        axes[2].scatter(vals, np.arange(1,len(vals)+1)/len(vals), s=8, color=color, alpha=.45)
    axes[2].set(xlabel='单帧检测耗时（毫秒）', ylabel='经验累积分布', ylim=(0,1.05), title='检测耗时分布：全部测试帧')
    axes[2].legend(frameon=False, fontsize=9, loc='lower right')
    polish(axes[2])
    fig.suptitle('气球检测开发测试：会话配对效果与处理成本', fontsize=15, y=.96)
    return finish(fig, output, '01_detector_paired_points',
        '左、中：每点为一个独立 UE 测试会话（每方法 n=2）；连线仅连接同一会话的两种检测器。\n'
        '右：每方法 48 帧，帧嵌套在两个会话内；CPU 预热后检测调用，不含采集或 VLX。仅支持当前受控场景。', records)


def age_plot(d, output, records):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    for i, session in enumerate(d['sessions']):
        label = '50毫米先导' if i==0 else '50毫米独立重复'
        pts = session['actual_target_points']
        color, marker = [(BLUE, 'o'), (TEAL, 'D')][i]
        for p in pts:
            assert abs(p['first_clear_margin_s']-(p['original_deadline_age_s']-p['first_clear_age_s'])) < 1e-8
            rows.append(dict(session_id=session['session_id'], **p))
        x = np.array([p['dispatch_age_s']*1000 for p in pts])
        y = np.array([p['first_clear_age_s']*1000 for p in pts])
        axes[0].scatter(x, y, s=21, alpha=.43, c=color, marker=marker, label=f'{label}（{len(pts)}点）')
        axes[1].scatter([p['slot'] for p in pts], [p['first_clear_margin_s']*1000 for p in pts],
                        s=16, alpha=.55, c=color, marker=marker, label=label)
        for p in pts:
            if p['reconstructed']:
                axes[0].scatter(p['dispatch_age_s']*1000, p['saved_clear_age_s']*1000,
                                marker='x', color=GRAY, s=55, zorder=5)
                axes[0].annotate('旧回执的收尾清零\n另属停止窗口',
                                 (p['dispatch_age_s']*1000, p['saved_clear_age_s']*1000),
                                 xytext=(210, 705), fontsize=9, color='#596B77',
                                 arrowprops=dict(arrowstyle='-', color=GRAY))
    save_csv(output/'target_command_age_points.csv', rows)
    axes[0].axhline(480, color=ORANGE, ls='--', lw=1, label='本批原源截止年龄480毫秒')
    axes[0].set(xlabel='目标指令派发时源图年龄（毫秒）', ylabel='首次目标清零时源图年龄（毫秒）',
                title='派发与首次清零：每点对应一个主帧', xlim=(100,285), ylim=(180,840))
    axes[0].legend(frameon=False, fontsize=8.5, loc='center left', bbox_to_anchor=(0,.64))
    axes[1].axhline(0, color=ORANGE, ls='--', lw=1)
    axes[1].set(xlabel='主测量槽编号（从0开始）', ylabel='首次目标清零距原截止余量（毫秒）',
                title='截止余量随测量推进的分布', ylim=(-10,280), xlim=(-10,405))
    axes[1].legend(frameon=False, fontsize=9, loc='lower left')
    for ax in axes: polish(ax)
    fig.suptitle('V11 固定高度开发验证：实际派发与首次目标清零', fontsize=15, y=.96)
    return finish(fig, output, '02_source_age_scatter',
        '只纳入有非零目标指令及停止回执的主帧：先导184点、重复185点，属于两个会话内的嵌套记录。\n'
        '圈点为首次目标清零；灰叉保留旧收尾窗口时刻。重复仍有采集中断与未知端点；清零不等于物理静止。', records)


def completeness_plot(d, output, records):
    ids = ['t1_process_zero_v11_20261007', 't1_process_50_v11_20261007', 't1_process_50_v11_rep1_20261007']
    selected = [next(r for r in d['runs'] if r['session_id']==sid) for sid in ids]
    summary, timing = [], []
    for r in selected:
        base = r['base_evidence']
        counts, grid = base['measured_identity_counts'], r['measured_grid_check']
        started, planned = base['measured_frame_count'], grid['planned_slots']
        assert sum(counts.values()) == started
        complete = base['measured_slot_status_counts'].get('CAPTURED',0)
        summary.append(dict(session_id=r['session_id'], correct=counts.get('CORRECT',0),
                            no_output=counts.get('NO_OUTPUT',0), unknown=counts.get('UNKNOWN_OUTPUT_STATE',0),
                            wrong=sum(v for k,v in counts.items() if k not in ['CORRECT','NO_OUTPUT','UNKNOWN_OUTPUT_STATE']),
                            not_started=planned-started, complete=complete, started=started, planned=planned,
                            net_xy_mm=None if r['endpoint']['net_xy_displacement_m'] is None else r['endpoint']['net_xy_displacement_m']*1000,
                            original_qualification=r['independent_scale_qualification']))
        for frame in r['frame_timing_rows']:
            for stage in ['frame_processing', 'historical_full']:
                t=frame['recorded_timing'][stage]
                timing.append(dict(session_id=r['session_id'], frame_id=frame['frame_id'], slot_index=frame['slot_index'],
                                   stage=stage, after_due_ms=None if t['after_due_s'] is None else t['after_due_s']*1000,
                                   status=t['status']))
    save_csv(output/'v11_session_summary.csv', summary)
    save_csv(output/'v11_frame_timing.csv', timing)
    fig, axes=plt.subplots(1,2,figsize=(12.8,5.7))
    names=['零门控','50毫米先导','50毫米独立重复']
    categories=[('correct','正确目标',TEAL),('no_output','无输出',ORANGE),
                ('unknown','输出未知','#AD98B5'),('wrong','其他身份状态','#C95750'),('not_started','未开始主槽','#DFE5E8')]
    left=np.zeros(3)
    for key,label,color in categories:
        widths=[r[key]/r['planned']*100 for r in summary]
        if any(widths):
            axes[0].barh(np.arange(3),widths,left=left,height=.43,color=color,label=label)
        left+=widths
    axes[0].set_yticks(range(3),names)
    axes[0].set(xlim=(0,105),ylim=(2.7,-.8),xlabel='占计划主槽比例（%）',title='身份与采集完整性：全部计划主槽')
    axes[0].legend(frameon=False,fontsize=8.5,loc='upper left',ncol=2)
    for i,r in enumerate(summary):
        endpoint='端点未知' if r['net_xy_mm'] is None else f"净位移 {r['net_xy_mm']:.3f} 毫米"
        axes[0].text(0,i+.37,f"完整/开始/计划：{r['complete']}/{r['started']}/{r['planned']}；{endpoint}",fontsize=8.7)
    for i,r in enumerate(selected[1:]):
        for stage,ls in [('frame_processing','-'),('historical_full','--')]:
            vals=np.sort([row['after_due_ms'] for row in timing if row['session_id']==r['session_id'] and row['stage']==stage and row['after_due_ms'] is not None])
            label=names[i+1]+('：帧处理' if stage=='frame_processing' else '：含停止的完成')
            axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=[BLUE,TEAL][i],ls=ls,lw=1.25,label=label)
    axes[1].axvline(400,color=ORANGE,ls=':',lw=1.2)
    axes[1].set(xlabel='相对原槽起点的完成跨度（毫秒）',ylabel='已知完成时刻的经验累积分布',
                title='处理完成与停止完成分别统计',ylim=(0,1.05),xlim=(0,1200))
    axes[1].legend(frameon=False,fontsize=8.3,loc='lower right')
    for ax in axes:polish(ax)
    fig.suptitle('V11 三次开发会话：分母、未知记录与整帧时效',fontsize=15,y=.96)
    return finish(fig,output,'03_completeness_and_timing',
        '重复：389帧完整、390帧已开始、400主槽计划；1帧输出未知、10槽未开始。初始化帧单独记账。\n'
        '右图先导各400个已知时刻，重复各389个；未知时刻不补零。400毫秒为原槽预算，独立尺度通过不代表整流程按时。',records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError('A_NEW_OUTPUT_DIRECTORY_IS_REQUIRED')
    data={key:json.loads(path.read_text(encoding='utf-8')) for key,path in INPUTS.items()}
    output.mkdir(parents=True)
    font,_=choose_font({ord(c) for c in '气球检测科研图表身份采集完整性时间'},None)
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,
                         'svg.fonttype':'path','axes.unicode_minus':True})
    records=[]
    audits=[detector_plot(data['detector'],output,records),age_plot(data['age'],output,records),
            completeness_plot(data['campaign'],output,records)]
    for p in output.glob('*.csv'):records.append({'path':str(p),'sha256':sha(p)})
    shutil.copyfile(__file__,output/Path(__file__).name)
    shutil.copyfile(ROOT/'python/plot_t1_identity_zh.py',output/'plot_t1_identity_zh.py')
    receipts=[]
    for key,path in INPUTS.items():
        dest=output/f'input_{key}.json'
        shutil.copyfile(path,dest)
        receipts.append({'role':key,'path':str(path),'sha256':sha(path),'snapshot':str(dest)})
    save_json(output/'artifact_receipt.json',dict(status='GENERATED_PENDING_VISUAL_REVIEW',
        inputs=receipts,files=records,font_layout_audits=audits,new_flights=0,cloud_requests=0,
        independent_detector_sessions=2,independent_v11_sessions=3,formal_trajectory_efficacy=False,
        runtime={'python':sys.version,'executable':sys.executable,'matplotlib':matplotlib.__version__,
                 'numpy':np.__version__},
        png_visual_review=False,pdf_visual_review=False,statistics='descriptive; no significance test or inferential CI',
        source_files=[{'path':str(p),'sha256':sha(p)} for p in [Path(__file__),ROOT/'python/plot_t1_identity_zh.py']]))
    print(json.dumps({'output':str(output),'figures':3,'csv_files':len(list(output.glob('*.csv')))},ensure_ascii=False))


if __name__=='__main__':main()

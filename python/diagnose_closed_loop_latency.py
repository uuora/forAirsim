"""Descriptive postflight latency decomposition; never drives the simulator."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from fixed_map_perception import read,save,sha,utc


def stats(values):
    values=[v for v in values if v is not None]
    if not values:return dict(n=0,median=None,p90=None,p99=None,max=None)
    return dict(n=len(values),median=float(np.median(values)),p90=float(np.percentile(values,90)),p99=float(np.percentile(values,99)),max=float(max(values)))


def diagnose(root):
    root=Path(root).resolve();output=root/'latency_diagnostic';output.mkdir(exist_ok=False)
    trial=root/'online_trial'
    published={}
    for row in read(trial/'online_publication_metrics.json'):
        published.setdefault(row['frame_id'],row)
    rows=[];groups=defaultdict(list)
    for path in sorted((trial/'online/frames').glob('*/perception.json')):
        frame=read(path)
        if 'processing_timings' not in frame:continue
        timing=frame['processing_timings'];capture=frame['capture'];checked=frame['processed_monotonic_s']
        scale=frame['semantic_frame']['detector_report'].get('conditional_scale_receipt',{})
        metric=published.get(frame['frame_id'])
        group=f"primary_{scale.get('primary_imgsz','unknown')}_fallback_{scale.get('fallback_used',False)}"
        row=dict(frame_id=frame['frame_id'],received_monotonic_s=capture['received_monotonic_s'],recorded_valid=frame['valid'],scale_path=group,
                 capture_archive_s=timing['receipt_to_capture_archive_s'],queue_s=timing['archive_to_inference_s'],inference_s=frame['inference_wall_s'],
                 final_check_age_s=checked-capture['received_monotonic_s'],
                 first_parent_age_s=metric['source_age_on_parent_receipt_s'] if metric else None,
                 final_check_to_first_parent_s=metric['received_by_parent_monotonic_s']-checked if metric else None)
        rows.append(row);groups[group].append(row)
    if not rows:raise ValueError('NO_MEASURED_PROCESSING_RECORDS')
    fields=['capture_archive_s','queue_s','inference_s','final_check_age_s','first_parent_age_s','final_check_to_first_parent_s']
    cfg=read(trial/'online_config.json');ttl=cfg['current_max_age_s']
    result=dict(utc=utc(),scope='Descriptive timing from the saved single simulator flight; missing publications remain missing, each frame uses its FIRST parent receipt. No causal attribution or generalized reliability claim.',script_sha256=sha(__file__),config=cfg['schema'],capture_period_s=cfg['capture_period_s'],freshness_threshold_s=ttl,processed_frames=len(rows),first_parent_receipts=sum(r['first_parent_age_s'] is not None for r in rows),first_parent_receipts_within_ttl=sum(r['first_parent_age_s'] is not None and r['first_parent_age_s']<=ttl for r in rows),metrics={field:stats([r[field] for r in rows]) for field in fields},scale_paths={key:dict(frames=len(values),inference=stats([r['inference_s'] for r in values]),first_parent_age=stats([r['first_parent_age_s'] for r in values])) for key,values in groups.items()},limitations=['Frame paths and scales depend on the image and previous resolution selection; groups are not randomized comparison arms.','Readback receipt ages are host timing, not hardware exposure synchronization.','System load was not controlled; the later prelaunch memory failure is a separate observation.','A fresh first parent receipt does not guarantee freshness at a later control command.'])
    save(output/'LATENCY_DIAGNOSTIC.json',result)
    with (output/'FRAME_TIMINGS.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=min(r['received_monotonic_s'] for r in rows)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    for field,label,color in [('final_check_age_s','At final validity check','#087e8b'),('first_parent_age_s','At first parent receipt','#b65f20')]:
        points=[r for r in rows if r[field] is not None]
        axes[0].scatter([r['received_monotonic_s']-start for r in points],[r[field] for r in points],s=5,alpha=.5,label=label,c=color)
    axes[0].axhline(ttl,color='#a33',ls='--',label='Existing freshness threshold')
    axes[0].set(ylabel='Source age (s)',title='Measured source age; rejected frames remain included');axes[0].legend(fontsize=8)
    for field,label in [('capture_archive_s','Capture archival work'),('queue_s','Wait after archive'),('inference_s','YOLO / candidate generation')]:
        axes[1].scatter([r['received_monotonic_s']-start for r in rows],[r[field] for r in rows],s=5,alpha=.45,label=label)
    axes[1].set(xlabel='Elapsed host time from first saved receipt (s)',ylabel='Measured stage duration (s)');axes[1].legend(fontsize=8)
    fig.suptitle(cfg['schema']+' single-flight diagnostic; no causal comparison')
    for ext in ['png','svg']:fig.savefig(output/('measured_latency.'+ext),dpi=180)
    plt.close(fig)
    text='# 飞行时延诊断\n\n仅描述本次原始计时。每帧使用第一次父进程收件；遗漏收件不补值，停止后的重复发布不计为新样本。\n\n|记录项|样本数|中位/s|P90/s|最大/s|\n|---|---:|---:|---:|---:|\n'
    labels=dict(capture_archive_s='收件后归档',queue_s='归档后排队',inference_s='推理及候选生成',final_check_age_s='最终判定帧龄',first_parent_age_s='父进程首次收件帧龄',final_check_to_first_parent_s='最终判定至父进程收件')
    for field in fields:
        m=result['metrics'][field]
        text+=f"|{labels[field]}|{m['n']}|{m['median']:.3f}|{m['p90']:.3f}|{m['max']:.3f}|\n" if m['n'] else f"|{labels[field]}|0|-|-|-|\n"
    text+=f"\n处理 {len(rows)} 帧，其中父进程首次收件帧龄≤{ttl} 秒的有 {result['first_parent_receipts_within_ttl']} 帧。这个数不是目标识别准确率，也不保证后续控制指令仍新鲜。\n\n分辨率分组受图像和历史分辨率选择影响，不是随机对照。系统负载未控制，不能据此断言采集周期或某一修复单独导致变化。后续内存拒绝单独记录，不能替代飞行期间内存证据。AI 使用说明：Codex 从保存的计时字段生成表格与图，无合成帧或成功记录。\n"
    (output/'README.md').write_text(text,encoding='utf-8')
    print(dict(processed_frames=len(rows),first_parent_receipts_within_ttl=result['first_parent_receipts_within_ttl'],metrics=result['metrics']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');diagnose(parser.parse_args().root)

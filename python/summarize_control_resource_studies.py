"""Read-only summaries of model-only budget diagnostics and negative results.

Does not launch an optimizer, simulator, or new experiment. All trials, including
cancelled workers and missing feasible results, remain in the exported CSV.
"""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
from run_control_wall_watchdog_research import read_incumbents

ROOT=Path(__file__).resolve().parents[1]
METHODS=('random','gaussian','coordinate','abom','abom_perturb','abom_perturb_screened')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def median(values):return statistics.median(values) if values else None


def summarize_study(folder,kind):
    completion=load(folder/'completion.json');rows=load(folder/'rows.json');precommit=load(folder/'precommit.json')
    if completion['status']!='COMPLETE' or completion['trials']!=len(rows) or len(rows)!=len(precommit['trials']):
        raise ValueError('study incomplete; cannot silently summarize a truncated matrix')
    inputs=[folder/n for n in ('completion.json','rows.json','precommit.json')];initials={};groups=defaultdict(list)
    normalized=[]
    for row in rows:
        index=row['index'];trial=row['trial'];case=Path(trial['case']).name
        if trial!=precommit['trials'][index]:raise ValueError('trial order differs from precommit')
        detail=folder/f'trial_{index:03}.json';inputs.append(detail)
        if kind=='watchdog':
            journal=folder/f'trial_{index:03}.jsonl';inputs.append(journal)
            recovered=read_incumbents(journal,trial['budget']);best=recovered['best']
            score=None if best is None else best['score']
            if score!=row['primary_best_feasible_score'] or recovered['completed_calls']!=row['completed_calls']:
                raise ValueError('durable journal disagrees with saved primary result')
            calls=[e['record'] for e in recovered['events'] if e['kind']=='COMPLETE']
            count=row['completed_calls'];overrun=row['parent_overrun_s']
        else:
            record=load(detail);calls=record['calls'];eligible=[r for r in calls if r['completed_by_deadline'] and r['feasible']]
            score=min((r['score'] for r in eligible),default=None)
            if score!=row['primary_best_feasible_score']:raise ValueError('soft-deadline raw calls disagree')
            count=row['full_oracle_calls'];overrun=row['overrun_s']
        # Compare the same first twenty evaluated points, not merely their seeds.
        key=(trial['case'],trial['budget'],trial['seed'])
        first=[r['parameters'] for r in calls[:20]]
        if len(first)==20:
            if key in initials and first!=initials[key]:raise ValueError('initial populations differ across methods')
            initials[key]=first
        item=dict(study=kind,case=trial['case'],method=trial['method'],seed=trial['seed'],budget_s=trial['budget'],
            status=row['status'],feasible=score is not None,score=score,full_completed_calls=count,overrun_s=overrun)
        normalized.append(item);groups[(trial['case'],trial['budget'],trial['method'])].append(item)
    summary=[]
    for (case,budget,method),items in groups.items():
        scores=[r['score'] for r in items if r['score'] is not None]
        summary.append(dict(case=case,budget_s=budget,method=method,n=len(items),feasible_n=len(scores),
            score_median=median(scores),score_min=min(scores) if scores else None,score_max=max(scores) if scores else None,
            completed_calls_median=median([r['full_completed_calls'] for r in items]),
            overrun_median_s=median([r['overrun_s'] for r in items]),overrun_max_s=max(r['overrun_s'] for r in items)))
    return normalized,summary,inputs


def summarize_screening(folder):
    summary=load(folder/'summary.json');pairs=summary['pairs'];inputs=[folder/'summary.json',folder/'precommit.json']
    if summary['status']!='COMPLETE':raise ValueError('incomplete screening experiment')
    for pair in pairs:
        a,b=[folder/f"case{pair['case_index']}_seed{pair['seed']}_screen{s}.json" for s in (0,1)];inputs.extend([a,b])
        left,right=load(a)['result'],load(b)['result']
        if [r['parameters'] for r in left['evaluations']]!=[r['parameters'] for r in right['evaluations']]:
            raise ValueError('screening changed proposal stream')
        if left['best_parameters']!=right['best_parameters'] or left['best_score']!=right['best_score']:
            raise ValueError('screening changed selected result')
    warm=[r['screened_wall_s']/r['baseline_wall_s'] for r in pairs if not r['cold_import_exposure']]
    baseline=sum(r['baseline_full_calls'] for r in pairs);screened=sum(r['screened_full_calls'] for r in pairs)
    return dict(pairs=len(pairs),baseline_full_calls=baseline,screened_full_calls=screened,
        saved_calls=baseline-screened,saved_fraction=(baseline-screened)/baseline,
        warm_wall_ratio_median=median(warm),warm_faster_pairs=sum(r<1 for r in warm),warm_pairs=len(warm)),inputs


def render_watchdog(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budgets=sorted({r['budget_s'] for r in summary});fig,axes=plt.subplots(1,len(budgets),figsize=(12,4.8),squeeze=False)
    for ax,budget in zip(axes[0],budgets):
        chosen=[next(r for r in summary if r['budget_s']==budget and r['method']==m) for m in METHODS]
        for i,r in enumerate(chosen):
            if r['score_median'] is not None:
                ax.errorbar(i,r['score_median'],yerr=[[r['score_median']-r['score_min']],[r['score_max']-r['score_median']]],fmt='o',capsize=3)
                ax.annotate(f"{r['feasible_n']}/{r['n']}",(i,r['score_median']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(len(METHODS)),['Random','Gaussian','Coordinate','ABOM','Paired','Paired+bound'],rotation=35,ha='right')
        ax.set_title(f'{budget:g}s fresh-process budget');ax.set_ylabel('Feasible surrogate score (lower is better)');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Fixed 8m target; median and observed range, 3 seeds; model only')
    fig.tight_layout();fig.savefig(output/'watchdog_quality.png',dpi=180);fig.savefig(output/'watchdog_quality.pdf');plt.close(fig)


def main(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    sources=dict(watchdog=ROOT/'reports/control_wall_watchdog_20261030_fixed_v1',
        soft=ROOT/'reports/control_wall_budget_20261030_two_cases_v1',
        screening=ROOT/'reports/control_target_screening_paired_20261030_v1')
    rows=[];summaries={};inputs=[]
    for kind in ('watchdog','soft'):
        r,s,p=summarize_study(sources[kind],kind);rows.extend(r);summaries[kind]=s;inputs.extend(p)
    screened,p=summarize_screening(sources['screening']);inputs.extend(p)
    before={str(p):sha(p) for p in inputs};output.mkdir(parents=True)
    save(output/'summary.json',dict(studies=summaries,screening=screened,airsim_calls=0))
    with (output/'all_trials.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 纯控制黑箱计算资源诊断（模型实验）','',
        '全部实验使用一阶滞后质点代理，AirSim调用为0。每个预算/方法/场景只有3个种子。以下为描述统计，不能推出真实飞行或普遍算法优越性。','',
        '## 有父进程截止器的固定8米实验','',
        '36次：6种搜索×2/5秒×种子13/31/47。进程启动、导入、初始化、训练、筛选和模型评价均消耗预算。保留日志中截止前已完成并发布的可行候选；父进程回收自身子进程并返回结果。操作系统回收超时另记，不声称硬实时。初始锚点已可行，不使用首次可行时间证明提效。','',
        '| 方法 | 预算s | 可行/总数 | 可行分数中位数 [最小,最大] | 完整调用中位数 | 返回超时中位数/最大值s |',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted(summaries['watchdog'],key=lambda r:(r['budget_s'],METHODS.index(r['method']))):
        quality='—' if r['score_median'] is None else f"{r['score_median']:.6f} [{r['score_min']:.6f},{r['score_max']:.6f}]"
        lines.append(f"| {r['method']} | {r['budget_s']:g} | {r['feasible_n']}/{r['n']} | {quality} | {r['completed_calls_median']:g} | {r['overrun_median_s']:.4f}/{r['overrun_max_s']:.4f} |")
    lines += ['', '神经适配器的冷启动开销在短预算下挤占评价次数；当前扰动和筛选未证明比低维坐标搜索更有效。该实验比较的是冷进程总开销，不是热运行神经算子的单步速度。截止取消为预定终止状态，不能删除或计作成功执行完固定评价数。','',
        '## 非抢占软截止诊断','',
        '72次，固定8米和种子1102目标。主要分数只取截止前完成的评价前缀，工作进程可能在库导入/训练完成后才返回，不能将此前缀分数称为按时返回的规划结果。初期少量试次与绘图诊断并行，存在CPU争用；仅保留为开发诊断，不能据此作严格运行速度结论。完整逐试次结果见CSV。','',
        '## 同容量筛选消融','',
        f"{screened['pairs']}个配对，候选流及所选参数完全相同；完整模型调用从{screened['baseline_full_calls']}降至{screened['screened_full_calls']}，只省{screened['saved_fraction']:.2%}。排除首次延迟导入暴露的配对后，筛选/基线时间比中位数{screened['warm_wall_ratio_median']:.3f}，{screened['warm_faster_pairs']}/{screened['warm_pairs']}配对更快。筛选本身有成本，暂不启用为默认改进。",'',
        '## 下一步与证据边界','',
        '优先保留常驻运行时、可回收截止和已完成最优解；神经权重仍按种子重新初始化，没有缓存目标值。后续需要在预先留出的目标及多个预算上做同硬件配对，再考察维数增加后神经开销能否被减少的模型调用抵消。筛选只对已声明的代价与边界成立，不是飞行安全证明。','',
        '所有失败、取消、未及时完成的评价保留。AI辅助实现与汇总；没有新增仿真、伪造计时或性能结论。']
    (output/'resource_diagnostics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    render_watchdog(output,summaries['watchdog'])
    if any(sha(Path(p))!=h for p,h in before.items()):raise RuntimeError('input changed during report')
    save(output/'provenance.json',dict(status='PASS',input_sha256=before,script_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},scope='Read-only recomputation; model-only diagnostics'))
    print(json.dumps(dict(status='PASS',rows=len(rows),output=str(output),screening=screened)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

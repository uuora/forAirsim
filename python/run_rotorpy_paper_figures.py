"""Recreate the paper's comparison figures from a frozen RotorPy campaign."""
from argparse import ArgumentParser
from html import escape
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from geometric_validation.paper_rotorpy_report import csv_write
from geometric_validation.rotorpy_results import load
from geometric_validation.runtime import ROOT, digest, write_json


METHODS = {'bezier': 'Bézier', 'hermite': 'Hermite', 'bspline': 'B-spline'}
ARMS = {
    'heading_baseline': ('Robust timing + heading', '#3274a1'),
    'velocity': ('Velocity TOPP', '#e1812c'),
    'collective': ('Velocity + thrust TOPP', '#3a923a'),
    'toppquad': ('TOPPQuad', '#c03d3e'),
}
COMPARATORS = tuple(ARMS)[:-1]
ROTOR_MAX_N = .14375
PAPER = 'https://arxiv.org/html/2309.11637v2#S5'


def grouped_boxes(ax, axes_labels, categories, values):
    group_width = .78
    step = group_width / len(categories)
    center = (len(categories)-1)/2
    for i, (category, color) in enumerate(categories):
        positions, selected = [], []
        for group, name in enumerate(axes_labels, 1):
            data = values(name, category)
            if len(data):
                positions.append(group+(i-center)*step)
                selected.append(data)
        if selected:
            ax.boxplot(selected, positions=positions, widths=step*.89, manage_ticks=False, patch_artist=True,
                       boxprops={'facecolor':color,'edgecolor':'#343434','linewidth':.8},
                       medianprops={'color':'#252525','linewidth':1.15},
                       whiskerprops={'color':'#454545','linewidth':.8}, capprops={'color':'#454545','linewidth':.8},
                       flierprops={'marker':'d','markersize':2.5,'markerfacecolor':'#555','markeredgewidth':0})
    ax.set_xlim(.48, len(axes_labels)+.52)
    ax.set_xticks(range(1, len(axes_labels)+1), [METHODS[name] for name in axes_labels])
    ax.grid(axis='y', color='#dedede', linewidth=.65)
    ax.set_axisbelow(True)


def legend(figure, names, *, top=.99):
    handles = [Patch(facecolor=ARMS[arm][1], edgecolor='#343434', linewidth=.65, label=ARMS[arm][0])
               for arm in names]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, top),
                  ncol=len(names), frameon=False, fontsize=9, handlelength=1.6)


def thrust_figure(methods, groups, names, *, feasible_only):
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 5.3), sharex=True)
    colors = [(arm, ARMS[arm][1]) for arm in names]
    for ax, field, label, bound in (
            (axes[0], 'reference_thrust_max_n', 'Maximum thrust [N]', ROTOR_MAX_N),
            (axes[1], 'reference_thrust_min_n', 'Minimum thrust [N]', 0.)):
        def values(method, arm):
            return [float(row[field]) for row in groups[method, arm] if row[field]
                    and (not feasible_only or row['reference_rotor_feasible'] == 'True')]
        grouped_boxes(ax, methods, colors, values)
        all_values = [number for m in methods for a in names for number in values(m, a)]
        low, high = min(min(all_values), bound), max(max(all_values), bound)
        delta = max(.004, (high-low)*.09)
        ax.set_ylim(low-delta, high+delta)
        if not feasible_only:
            ax.axhspan(0., ROTOR_MAX_N, color='#dce9da', alpha=.50, zorder=0)
        ax.axhline(bound, color='#376344', linestyle='--', linewidth=.9)
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    legend(fig, names)
    fig.tight_layout(rect=(.01, .01, .995, .94))
    return fig


def reference_time_figure(methods, groups):
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    colors = [(arm, color) for arm, (_, color) in ARMS.items()]
    grouped_boxes(ax, methods, colors,
                  lambda m, a: [float(r['reference_s']) for r in groups[m,a]
                                if r['reference_rotor_feasible'] == 'True' and r['reference_s']])
    ax.set_ylabel('Reference traversal time [s]')
    legend(fig, ARMS)
    fig.tight_layout(rect=(.01, .01, .995, .93))
    return fig


def reduction_figure(methods, paired):
    samples = {(m,a): [float(p['reference_reduction_pct']) for p in paired
                       if p['method']==m and p['comparator']==a and p['motor_feasible']=='True'
                       and p['reference_reduction_pct']] for a in COMPARATORS for m in methods}
    all_values = [x for values in samples.values() for x in values]
    lower = int(np.floor(min(min(all_values),0.)/5)*5)
    upper = int(np.ceil(max(max(all_values),0.)/5)*5)
    bins = np.arange(lower, upper+5, 5.)
    fig, axes = plt.subplots(len(COMPARATORS), 3, figsize=(9.5, 7.1), sharex=True, sharey=True, squeeze=False)
    for i, arm in enumerate(COMPARATORS):
        for j, method in enumerate(methods):
            ax = axes[i,j]
            values = samples[method,arm]
            ax.hist(values, bins=bins, color=ARMS[arm][1], edgecolor='#444444', linewidth=.55)
            ax.axvline(0., color='#333333', linestyle='--', linewidth=.9)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#e4e4e4', linewidth=.55)
            if i == 0:
                ax.set_title(METHODS[method], fontweight='medium')
            if j == 0:
                ax.set_ylabel(ARMS[arm][0]+'\nRoutes')
            ax.text(.97,.94, f'n = {len(values)}', transform=ax.transAxes, va='top', ha='right', fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.supxlabel('Paired reference time reduction [%]', fontsize=11)
    fig.tight_layout(rect=(.014,.035,.997,.997))
    return fig


def planning_stages(root, runs):
    by_key = {(r['route'],r['method'],r['arm']):r for r in runs}
    stages = []
    digest_items = {}
    for route, method, arm in by_key:
        if arm != 'toppquad':
            continue
        path = root/route/method/arm/'raw_solution.json'
        upstream = load(path)['raw_times_s'][-1]
        current = float(by_key[route,method,arm]['reference_s'])
        speed_topp = float(by_key[route,method,'velocity']['reference_s'])
        stages.append(dict(route=route, method=method, speed_topp_reference_s=speed_topp,
                           upstream_discrete_s=upstream, executed_reference_s=current,
                           upstream_minus_speed_topp_s=upstream-speed_topp,
                           rebuild_delta_s=current-upstream,
                           executed_minus_speed_topp_s=current-speed_topp))
        digest_items[path.relative_to(root).as_posix()] = digest(path)
    return stages, digest_items


def stage_figure(methods, stages):
    definitions = [('upstream_minus_speed_topp_s','Upstream discrete\nminus velocity TOPP','#3274a1'),
                   ('rebuild_delta_s','Executed reference\nminus upstream discrete','#e1812c'),
                   ('executed_minus_speed_topp_s','Executed reference\nminus velocity TOPP','#c03d3e')]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.9), sharey=True)
    for ax,(field,title,color) in zip(axes,definitions,strict=True):
        grouped_boxes(ax, methods, [('metric',color)],
                      lambda m, _: [row[field] for row in stages if row['method']==m])
        ax.set_title(title)
        ax.axhline(0., color='#444444', linewidth=.8, linestyle='--')
    axes[0].set_ylabel('Paired time difference [s]')
    fig.tight_layout(rect=(.01,.01,.996,.99))
    return fig


def export(fig, output, name, pdf):
    fig.savefig(output/f'{name}.png', dpi=280, facecolor='white')
    fig.savefig(output/f'{name}.svg', facecolor='white')
    pdf.savefig(fig)
    plt.close(fig)


def write_page(output, figures, stages):
    chart_markup = []
    for name, title, caption in figures:
        chart_markup.append(f'''<figure>
  <h2>{escape(title)}</h2>
  <img src="{name}.svg" alt="{escape(title)}" loading="lazy">
  <figcaption>{escape(caption)} <a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></figcaption>
</figure>''')
    time_before = median(row['upstream_minus_speed_topp_s'] for row in stages)
    rebuild = median(row['rebuild_delta_s'] for row in stages)
    time_after = median(row['executed_minus_speed_topp_s'] for row in stages)
    page = f'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RotorPy 论文式对比图</title>
<style>
  html {{ background:#f3f5f5; color:#152329; font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif; }}
  main {{ max-width:940px; margin:auto; padding:24px 16px 48px; }}
  header,figure {{ margin:0 0 16px; padding:clamp(16px,3vw,28px); border:1px solid #d9e0e0; border-radius:10px; background:#fff; }}
  h1,h2 {{ margin:0 0 8px; line-height:1.3; }}
  h2 {{ font-size:1.2rem; }}
  p {{ margin:6px 0 10px; }}
  img {{ width:100%; height:auto; display:block; margin:12px 0; }}
  a {{ color:#1d5275; text-underline-offset:.16em; }}
</style>
<main>
 <header>
  <h1>RotorPy 论文式对比图</h1>
  <p>100 条共享航路、三种路径构造、四种方法，共 1,200 条首飞记录。</p>
  <p><a href="../#comparisonSection">返回实验看板</a> · <a href="paper_figures.pdf">下载整套 PDF</a> ·
     <a href="planning_stage_times.csv">计时阶段数据</a> · <a href="manifest.json">绘图来源</a></p>
  <p>图 2–5 沿用原论文的箱线图和直方图口径，采用本实验的稳健计时、速度 TOPP、
     速度＋总推力 TOPP 和 TOPPQuad。 <a href="{PAPER}">TOPPQuad 原论文第 V 节</a></p>
  <p>相对速度 TOPP，上游 TOPPQuad 离散时间的配对中位数差为 {time_before:+.2f} 秒；
     连续参考比离散时间多 {rebuild:.2f} 秒；连续参考相对速度 TOPP 为 {time_after:+.2f} 秒。
     三项各自计算中位数。连续重优化采用 4.2 m/s 规划限速，以及加速度、倾角和电机余量约束。</p>
 </header>
 {''.join(chart_markup)}
</main>
</html>
'''
    (output/'index.html').write_text(page, encoding='utf-8')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'reports/rotorpy_current')
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    design = load(root/'design.json')
    methods = design['methods']
    if tuple(design['arms']) != tuple(ARMS) or tuple(methods) != tuple(METHODS):
        raise ValueError('Expected the current four-arm, three-construction comparison')
    runs, paired = (load(root/file) for file in ('runs.csv','paired.csv'))
    n = len(design['routes'])
    run_keys = {(r['route'],r['method'],r['arm']) for r in runs}
    pair_keys = {(r['route'],r['method'],r['comparator']) for r in paired}
    expected = {(r['id'],m,a) for r in design['routes'] for m in methods for a in ARMS}
    if len(runs) != len(run_keys) or run_keys != expected or len(paired) != len(pair_keys) or pair_keys != {
            key for key in expected if key[2] != 'toppquad'}:
        raise ValueError('The planned route/method/mode matrix is incomplete or duplicated')
    groups = {(m,a): [r for r in runs if (r['method'],r['arm']) == (m,a)] for m in methods for a in ARMS}
    stages, source_hashes = planning_stages(root, runs)
    output = root/'paper_figures'
    output.mkdir(exist_ok=True)
    descriptions = [
        ('fig02_baseline_thrust', '图 2 · 三种对照方法的推力范围',
         f'每个箱体为 {n} 条参考轨迹。上排是每条轨迹所有电机和时刻的推力最大值，下排是最小值。浅绿为允许区间。'),
        ('fig03_feasible_thrust', '图 3 · 加入 TOPPQuad 的推力范围',
         '仅纳入参考推力合规的轨迹。虚线分别为 0.14375 N 和 0 N。'),
        ('fig04_reference_time', '图 4 · 参考飞行时间分布',
         '仅纳入参考逐电机推力合规的样本，按三种路径构造分别比较四组。'),
        ('fig05_paired_reduction', '图 5 · 同路径参考时间变化',
         '每格按相同航路和路径构造配对，双方参考推力均合规。横轴为 (对照用时 − TOPPQuad 用时) / 对照用时，负值表示 TOPPQuad 更慢。'),
        ('diagnostic_time_stages', '补充图 · TOPPQuad 的三个时间差',
         '对同航路分别计算离散解与速度 TOPP、连续参考与离散解、连续参考与速度 TOPP 的时间差。离散解为规划中间结果，连续参考是闭环所跟踪的输入。'),
    ]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'path'}):
        with PdfPages(output/'paper_figures.pdf', metadata={'Title':'RotorPy paper-style comparison figures'}) as pdf:
            figures = (thrust_figure(methods, groups, COMPARATORS, feasible_only=False),
                       thrust_figure(methods, groups, tuple(ARMS), feasible_only=True),
                       reference_time_figure(methods, groups),
                       reduction_figure(methods, paired), stage_figure(methods, stages))
            for (name, _, _), fig in zip(descriptions, figures, strict=True):
                export(fig, output, name, pdf)
    csv_write(output/'planning_stage_times.csv', stages)
    write_page(output, descriptions, stages)
    write_json(output/'manifest.json', dict(
        paper=PAPER, routes=n, runs=len(runs), pairs=len(paired), motor_bound_n=[0.,ROTOR_MAX_N],
        source_sha256={file:digest(root/file) for file in ('design.json','runs.csv','paired.csv')},
        raw_solution_sha256=source_hashes, source_program_sha256=digest(Path(__file__)),
        figures=[dict(name=name, title=title, caption=caption) for name,title,caption in descriptions],
    ))
    print(output/'index.html')


if __name__ == '__main__':
    main()

"""Freeze compact public evidence and scientific figures for budget studies."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'reports/abom_budget_20261029'
PUBLIC=ROOT/'experiments/abom_budget_20261029'
FIGURES=ROOT/'docs/research-hermite-20261027/figures/abom_budget'


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    PUBLIC.mkdir(parents=True,exist_ok=False)
    FIGURES.mkdir(parents=True,exist_ok=True)
    index={}
    for name in ['geometry_main','geometry_closed_form','control_cached_pilot','dropout_calibration']:
        source=LOCAL/name
        assert load(source/'audit.json')['status']=='PASS'
        target=PUBLIC/name
        target.mkdir()
        for p in source.glob('*.json'):
            if p.name in ['manifest.json','summary.json','rows.json','audit.json','completion.json','paired_comparisons.json']:
                shutil.copy2(p,target/p.name)
        archive=PUBLIC/f'{name}_raw.zip'
        with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(source.rglob('*')):
                if p.is_file(): z.write(p,arcname=str(p.relative_to(source)).replace('\\','/'))
        with ZipFile(archive) as z:
            assert z.testzip() is None
            for p in source.rglob('*'):
                if p.is_file(): assert hashlib.sha256(z.read(str(p.relative_to(source)).replace('\\','/'))).digest()==hashlib.sha256(p.read_bytes()).digest()
        index[archive.name]=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                entries=len(ZipFile(archive).namelist()),status='CRC_AND_ALL_SOURCE_BYTES_VERIFIED')
    diagnostics=PUBLIC/'excluded_diagnostics.zip'
    with ZipFile(diagnostics,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
        for name in ['geometry_smoke','control_pilot']:
            for p in sorted((LOCAL/name).rglob('*')):
                if p.is_file(): z.write(p,arcname=str(p.relative_to(LOCAL)).replace('\\','/'))
    index[diagnostics.name]=dict(bytes=diagnostics.stat().st_size,sha256=hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        reason='Smoke has lazy-start timing effect; first control pilot aborted due scenario-caching mismatch. Excluded from timing conclusions.')
    shutil.copy2(LOCAL/'regression_tests.txt',PUBLIC/'regression_tests.txt')
    (PUBLIC/'bundle_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    plot()
    print(json.dumps(index,indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    groups=[(f,l) for f in ['single_bump','paired_bump'] for l in ['central','early_offset','two_obstacles']]
    methods=['abom','abom_perturb','gaussian']
    labels=['ABOM adapter','ABOM + perturbation','Gaussian']
    colors=['#24628f','#d47b36','#4d8e68']
    rows=[]
    fig,axs=plt.subplots(1,2,figsize=(11.5,4),sharey=True)
    for ax,campaign,title in zip(axs,['geometry_main','geometry_closed_form'],['Numerical quality bound','Closed-form quality bound']):
        data=load(PUBLIC/campaign/'summary.json')
        for j,(m,label,color) in enumerate(zip(methods,labels,colors)):
            selected=[next(r for r in data if (r['family'],r['layout'],r['method'])==(f,l,m)) for f,l in groups]
            y=np.array([r['median_wall_ratio'] for r in selected])
            ci=np.array([r['exploratory_paired_bootstrap95'] for r in selected])
            x=np.arange(6)+(j-1)*.22
            ax.errorbar(x,y,yerr=np.vstack((y-ci[:,0],ci[:,1]-y)),fmt='o',capsize=3,color=color,label=label)
            rows.extend(dict(campaign=campaign,**r) for r in selected)
        ax.axhline(1.,color='#777777',ls='--',lw=1)
        ax.set_xticks(range(6),['4D\ncentral','4D\noffset','4D\ntwo obs.','7D\ncentral','7D\noffset','7D\ntwo obs.'])
        ax.set_title(title)
        ax.set_xlabel('Fixed development condition (10 paired seeds each)')
    axs[0].set_ylabel('Screened / original wall time (lower is faster)')
    handles,legend_labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Identical proposals and selected solutions; warm-process timing\nPoints: group medians; bars: exploratory 95% paired bootstrap intervals',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.9))
    for extension in ['png','pdf']: fig.savefig(FIGURES/f'exact_screening.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (FIGURES/'screening_data.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['campaign','family','layout','method','n','median_wall_ratio','calls_saved_fraction']
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    raw=load(PUBLIC/'dropout_calibration/rows.json')
    fig,axs=plt.subplots(2,3,figsize=(10,6),sharey=True)
    for ax,(f,l) in zip(axs.flat,groups):
        data=[[r['best_score'] for r in raw if (r['family'],r['layout'],r['dropout'])==(f,l,p)] for p in [.95,.5,.1]]
        box=ax.boxplot(data,labels=['.95','.5','.1'],patch_artist=True,showfliers=True)
        for patch,color in zip(box['boxes'],colors): patch.set_facecolor(color);patch.set_alpha(.55)
        ax.set_title(('4D' if f=='single_bump' else '7D')+' / '+l)
        ax.set_xlabel('Dropout (10 shared initialization seeds)')
    for ax in axs[:,0]: ax.set_ylabel('Selected geometry score (lower is better)')
    fig.suptitle('ABOM parameter calibration: 256 complete evaluations, no screening',fontsize=11)
    fig.tight_layout()
    for extension in ['png','pdf']: fig.savefig(FIGURES/f'dropout_calibration.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__': main()

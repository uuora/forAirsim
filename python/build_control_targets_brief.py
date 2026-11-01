"""One-page progress brief from the real, audited protocol12 six-group run."""
import argparse,hashlib,json,sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,SimpleDocTemplate,Spacer,Table,TableStyle
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    source=ROOT/'reports/control_targets_fixed_v12_comparison/summary.json'
    summary=json.loads(source.read_text(encoding='utf-8'));rows=summary['records']
    run=ROOT/'reports/airsim_control_targets_v12_fixed_six/replay'
    original=json.loads((run/'rows.json').read_text(encoding='utf-8'))
    if len(rows)!=6 or len(original)!=6 or not all(r['metrics']['control_error_gates_passed'] for r in original):
        raise ValueError('expected six valid control-passing records')
    font='C:/Windows/Fonts/simhei.ttf';pdfmetrics.registerFont(TTFont('CN',font))
    body=ParagraphStyle('body',fontName='CN',fontSize=9.3,leading=14.3,spaceAfter=5,wordWrap='CJK',textColor=colors.HexColor('#263849'))
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11.5)
    title=ParagraphStyle('title',parent=body,fontSize=19,leading=25,spaceAfter=9,textColor=colors.HexColor('#17394D'))
    heading=ParagraphStyle('heading',parent=body,fontSize=11,leading=16,spaceBefore=7,spaceAfter=4,textColor=colors.HexColor('#176E88'))
    cell=ParagraphStyle('cell',parent=small,spaceAfter=0)
    p=lambda s,style=body:Paragraph(s,style)
    story=[p('研究内容一：纯控制仿真进展',title),
        p('2026-10-30 | 03三维双目标地图 | 无障碍物 | 协议12 | 开发先导',small),
        p('问题：固定跟踪控制器，通过曲线参数及执行时长优化，能否减少规划计算量并保持可执行性与跟踪精度？当前已接通“选择一种运行、固定/随机目标、勾选历史对比”。'),
        p('已实现的闭环系统',heading),
        p('四维参数(dₓ,dᵧ,d_z,T) → 简化模型搜索 → 冻结参考 → AirSim真实执行 → 独立日志重算。五次基线加入端状态保持形变后至多六次；Bézier/Hermite可表示同一物理轨迹。搜索不以AirSim为黑箱。'),
        p('统一速度外环：q̇ + 2(q-p) + 0.8(q̇-v) + 0.9q̈，速度指令上限4m/s、变化率4m/s²。按实际状态时间戳同步参考；每次独立重置、仿真初态p/v/a读回核验。控制器标定后冻结，门槛未放宽。'),
        p('固定8米目标的实际执行结果：每组1次',heading)]
    names=['五次Bézier基线','五次Hermite基线','Bézier + 基础ABOM','Hermite + 基础ABOM','Bézier + 扰动候选','Hermite + 扰动候选']
    ids=['bezier_ordinary','hermite_ordinary','bezier_abom','hermite_abom','bezier_abom_improved_candidate','hermite_abom_improved_candidate']
    by_id={r['group_id']:r for r in original};data=[['方法','执行T/s','RMSE/m','峰值/m','终端位置/m']]
    for name,key in zip(names,ids):
        m=by_id[key]['metrics'];data.append([name,*[f'{m[k]:.4f}' for k in ['reference_duration_s','position_rmse_m','max_position_error_m','terminal_position_error_m']]])
    table=Table([[p(t,cell) for t in row] for row in data],colWidths=[181,76,82,82,96])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EEF2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F4F7F9'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story += [table,Spacer(1,6),p('六组通过原误差门槛：峰值≤0.5m、终端位置≤0.25m、终端速度≤0.5m/s。优化组更快但误差更高；Hermite综合分数基线0.44082、基础0.45178、候选0.45438（越小越好），尚无整体改善证据。',small),
        p('计算效率：负结果也保留',heading),
        p('ABOM研究适配器使用种群20、候选80；候选方法以成对局部参数扰动替换一半子代，不是原论文完整复现。36次2/5秒冷启动预算实验中，低维坐标搜索的代理质量更好。9个同容量筛选配对仅节省10/720次完整模型调用，热计时中位数反增14.5%，暂不启用默认筛选。'),
        p('工程改进：常驻进程仅复用导入环境，一次配对总规划12.02→8.18秒；重新初始化网络，仍做326次完整模型调用。单次工程测量不作为算法创新或普遍提效结论。',small),
        p('下一步：先解释模型与AirSim的质量排序反转',heading),
        p('补仅时长/仅形状、几何/闭环评价消融；冻结新目标和配对种子，再做初态偏差、模型失配与外扰。主要指标为固定墙钟内可返回的最佳可行质量，补充等候选数，所有失败/超时保留。'),
        p('证据：reports/airsim_control_targets_v12_fixed_six；control_targets_fixed_v12_comparison；control_resource_diagnostics_20261030_v1。详细参数、任务书及原始状态/指令/事件均已保存。',small),
        p('边界：开发小样本，无真机/姿态轨迹/完整高敏捷验证；速度指令限幅不是电机饱和。AI辅助实现与整理。开题依据：原报告§2.2.1。',small)]
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=39,rightMargin=39,topMargin=32,bottomMargin=31,title='研究内容一纯控制仿真进展')
    def footer(canvas,doc):
        canvas.setFont('CN',7.5);canvas.setFillColor(colors.HexColor('#617482'));canvas.drawString(39,17,'导师讨论稿 / 已执行结果与后续计划分别标识');canvas.drawRightString(A4[0]-39,17,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(str(output)).pages)
    if pages!=1:raise RuntimeError(f'expected one page, got {pages}')
    output.with_suffix('.provenance.json').write_text(json.dumps(dict(pages=pages,comparison_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_rows_sha256=hashlib.sha256((run/'rows.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',pages=pages,output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);build(parser.parse_args().output)

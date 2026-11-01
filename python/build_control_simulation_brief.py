"""Render a one-page Chinese research brief from an audited E0 campaign.

Use the bundled document Python runtime with reportlab/pypdf installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def build(campaign, output):
    campaign, output = Path(campaign), Path(output)
    audit = json.loads((campaign / "audit.json").read_text(encoding="utf-8"))
    summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((campaign / "rows.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["rollouts"] != 24:
        raise ValueError("Expected the audited 24-rollout E0 protocol")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a delivered PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    font = Path("C:/Windows/Fonts/simhei.ttf")
    pdfmetrics.registerFont(TTFont("BriefCN", str(font)))
    normal = ParagraphStyle("body", fontName="BriefCN", fontSize=9.4, leading=14.2,
                            textColor=colors.HexColor("#263447"), spaceAfter=6,
                            wordWrap="CJK", alignment=TA_LEFT)
    title = ParagraphStyle("title", parent=normal, fontSize=18, leading=24,
                           textColor=colors.HexColor("#15354B"), spaceAfter=7)
    heading = ParagraphStyle("heading", parent=normal, fontSize=10.7, leading=16,
                             textColor=colors.HexColor("#176C79"), spaceBefore=5, spaceAfter=4)
    small = ParagraphStyle("small", parent=normal, fontSize=8.0, leading=11.5, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=normal, fontSize=8.8, leading=13, spaceAfter=0)
    doc = SimpleDocTemplate(str(output), pagesize=A4, topMargin=37, bottomMargin=33,
                            leftMargin=39, rightMargin=39,
                            title="研究内容一控制仿真实验方案", author="AI-assisted research design")
    p = lambda text, style=normal: Paragraph(text, style)
    story = [p("研究内容一：控制仿真实验方案", title),
             p("2026-10-29  |  固定控制器的轨迹优化  |  E0最小验证已完成", small),
             p("研究问题：在同一起终状态、固定跟踪控制器及执行约束下，改进ABOM能否以更少的规划计算时间和闭环仿真调用，获得同等或更好的可执行轨迹？"),
             p("曲线参数与时长 → 时序参考 → 固定控制器 → 仿真执行 → 误差评价 → ABOM更新", heading),
             p("先用三维一阶速度滞后质点进行开发，再接入AirSim复核。第一轮优化三维形状扰动和时长T；固定端点位置/速度/加速度、速度前馈加位置反馈控制器及限幅。Bézier/Hermite在等价参数下先做表达核验。"),
             p("实验安排", heading)]
    cells = [
        ["实验", "对照与主要固定条件", "主要问题/指标"],
        ["E0", "直线、绕行、高度变化；T=5.5/3秒；四个单因素条件", "时间对齐、控制记录、误差和限幅核验"],
        ["E1", "初始参考、几何目标优化、闭环目标优化；固定ABOM", "闭环反馈是否改善实际跟踪"],
        ["E2", "随机、高斯、坐标、校准ABOM、改进ABOM；同预算", "预算内可输出率、最佳质量、完整调用及耗时"],
        ["E3", "基础、仅筛选、仅反馈、组合；零/打乱特征对照", "收益来源、容量影响与额外开销"],
        ["E4", "冻结结果后改变初态/响应；再做AirSim复核", "扰动下跟踪效果、迁移和执行失败"],
    ]
    table = Table([[p(t, cell) for t in row] for row in cells], colWidths=[36, 253, 228], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E4EEF2")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F8FAFB"), colors.white]),
        ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#9CB3BF")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([table, Spacer(1,7),
        p("公平比较与正式规模", heading),
        p("以固定墙钟预算内的最佳可行质量为主，等候选数为辅；分别报告规划计算时间、飞行时长、RMSE、峰值/终端误差、净空及指令限幅。初始化、训练和必要验证计时，失败与超时保留。"),
        p("正式E2拟用3类新场景×2档难度×20配对种子×5方法=600次/制度。三个墙钟截点可复用600次anytime运行；等候选实验另600次。场景坐标、最终预算B与退化容限仍待冻结。", small),
        p("本轮实际验证", heading),
        p(f"已执行{summary['rollouts']}次简化闭环：5.5秒组12条均通过控制误差门槛，3秒组12条均未通过，快速工况出现持续指令限幅。独立核查PASS，覆盖{audit['state_samples']:,}条状态和{audit['command_intervals']:,}个控制区间；六组基底核验通过。"),
    ])
    slow = [r["position_rmse_m"] for r in rows if r["condition_id"] == "nominal" and r["duration_s"] == 5.5]
    fast = [r["position_rmse_m"] for r in rows if r["condition_id"] == "nominal" and r["duration_s"] == 3.]
    story.extend([
        p(f"三类名义参考位置RMSE：5.5秒为{min(slow):.3f}-{max(slow):.3f} m，3秒为{min(fast):.3f}-{max(fast):.3f} m。12/24是固定样例覆盖计数，不是总体成功概率；本批未调用优化器。", small),
        p("下一步：统一三维AirSim执行、初态/时钟及结束规则，补完整墙钟预算接口，再做E1与机制消融。AirSim安装文件存在，本轮默认RPC未连接，尚未执行AirSim验证。", normal),
        p("证据边界：简化模型不含机体、电机和碰撞；指令限幅不代表电机饱和。既有诊断反馈开发结果尚无稳定优势。本文由AI辅助整理，正式优化实验为待验证计划。", small),
        p("依据：开题§2.2.1/§2.3.2；ABOM arXiv:2601.21475v2 §3；AirSim Simple Flight文档；本地E0 config、summary、audit及完整任务书。", small),
    ])
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B9C8CF"))
        canvas.line(39,28,A4[0]-39,28)
        canvas.setFont("BriefCN",7.7)
        canvas.setFillColor(colors.HexColor("#617482"))
        canvas.drawString(39,16,"研究内容一 / 导师讨论稿 / 设计与已执行证据分别标识")
        canvas.drawRightString(A4[0]-39,16,str(document.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    count = len(PdfReader(str(output)).pages)
    if count != 1:
        raise RuntimeError(f"Expected one page, rendered {count}")
    print(json.dumps(dict(output=str(output.resolve()),pages=count),ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    build(args.campaign,args.output)

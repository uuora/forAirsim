"""Descriptive counts from the frozen test receipt; no independence inference."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run():
    cache=ROOT/"logs/t1_balloon_model_runtime/matplotlib";cache.mkdir(parents=True,exist_ok=True)
    os.environ["MPLCONFIGDIR"]=str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report_path=ROOT/"logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"]!="PASS" or report["unique_test_frames"]!=48 or report["independent_test_episodes"]!=2:
        raise ValueError("Frozen complete pilot test required")
    output=ROOT/"logs/t1_balloon_adaptation_figures_v1p2_20261004";output.mkdir(exist_ok=False)
    a=report["summary_by_model"]["original_COCO_sports_ball"]
    b=report["summary_by_model"]["specialist_balloon"]
    fig,axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={"width_ratios":[1.7,1]})
    x=np.arange(3);colors=["#66758C","#287E86"]
    for index,(row,label) in enumerate(((a,"Original COCO"),(b,"Balloon specialist"))):
        values=[row["true_positive"],row["false_negative"],row["false_positive"]]
        bars=axes[0].bar(x+(index-.5)*.36,values,.36,color=colors[index],label=label)
        axes[0].bar_label(bars,padding=3,fontsize=10)
    axes[0].set_xticks(x,["Matched objects","Missed objects","Extra boxes"])
    axes[0].set_ylabel("Count");axes[0].set_ylim(0,65)
    axes[0].set_title("56 visible balloon labels");axes[0].legend(frameon=False,fontsize=9)
    bars=axes[1].bar([0,1],[a["both_balloons_detected_frames"],b["both_balloons_detected_frames"]],color=colors,width=.6)
    axes[1].bar_label(bars,padding=3,fontsize=10)
    axes[1].set_xticks([0,1],["Original COCO","Balloon specialist"],fontsize=9)
    axes[1].set_ylabel("Frames with both balloons detected");axes[1].set_ylim(0,19)
    axes[1].set_title("16 two-balloon frames")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Controlled detector pilot: 2 held-out UE sessions, 48 frames",fontsize=12)
    fig.text(.5,.015,"Same map, mesh and red material; frames are nested within sessions. Confidence 0.25; matching IoU 0.5.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.045,1,.94])
    artifacts=[]
    for extension in ("png","pdf"):
        path=output/("detector_comparison."+extension);fig.savefig(path,dpi=200)
        artifacts.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    receipt={"status":"PASS","source_report":str(report_path),"source_report_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"artifacts":artifacts,
        "ai_assistance":"AI assisted plotting; bars are deterministic counts from archived evaluation"}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","figure":artifacts[0]["path"]}))


if __name__=="__main__":run()

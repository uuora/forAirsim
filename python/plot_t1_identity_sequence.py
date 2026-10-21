"""Receipt-clock timelines and descriptive timing counts; no trial inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(report_paths, output):
    cache=ROOT/"logs/t1_balloon_model_runtime/matplotlib"
    cache.mkdir(parents=True,exist_ok=True)
    os.environ["MPLCONFIGDIR"]=str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    reports=[json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    artifacts=[]
    colors={"CORRECT":"#247F79","WRONG":"#C34343","PENDING":"#C39936",
        "NO_OUTPUT":"#728091","UNRESOLVED":"#835DA7"}
    for version,report in enumerate(reports,1):
        fig,axes=plt.subplots(len(report["episodes"]),1,figsize=(10,7),sharex=True)
        for index,(ax,episode) in enumerate(zip(axes,report["episodes"])):
            rows=episode["rows"]
            origin=next((row["planned_start_host_s"]-row["planned_start_relative_s"]
                for row in rows if row["planned_start_host_s"] is not None),None)
            ax.axvspan(5,7,color="#D4DFE6",alpha=.55)
            if origin is not None:
                for label,color in colors.items():
                    points=[row for row in rows if row["identity_point_label"]==label
                        and row["process_completed_host_s"] is not None]
                    ax.scatter([row["process_completed_host_s"]-origin for row in points],
                        [1]*len(points),s=26,color=color,zorder=4)
                requests=episode["submitted_requests"]
                ax.scatter([request["submitted_monotonic_s"]-origin for request in requests],
                    [1.45]*len(requests),marker="^",s=24,color="#28527A",zorder=3)
                raw=json.loads(Path(episode["report_path"]).read_text(encoding="utf-8"))
                for phase in raw["phase_transition_records"][1:]:
                    ax.axvline(phase["setup_completed_host_monotonic_s"]-origin,
                        color="#99A7B4",ls="--",lw=.8)
            missed=[row for row in rows if not row["captured"]]
            ax.scatter([(row["planned_start_relative_s"]+row["planned_end_relative_s"])/2
                for row in missed],[.15]*len(missed),s=22,color="#7E8993",marker="x")
            condition="Striped target" if episode["recovery_observability"]=="VISUAL_ATTRIBUTE" else "Identical targets"
            ax.set_title(f"{condition}, session {index%2+1}: {episode['captured_slots']}/{episode['planned_slots']} captured",loc="left",fontsize=10)
            ax.set_yticks([.15,1,1.45],["Missing slot","Decision","VLX request"],fontsize=8)
            ax.set_ylim(-.15,1.8)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.grid(axis="x",alpha=.15)
        axes[-1].set_xlabel("Host seconds since schedule start (not exposure time)")
        axes[-1].set_xlim(0,12.6)
        legend=[Line2D([0],[0],marker="o",color="none",markerfacecolor=color,label=label.replace("_"," "),markersize=6)
            for label,color in colors.items() if any(row["identity_point_label"]==label
                for ep in report["episodes"] for row in ep["rows"])]
        fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=4,frameon=False,fontsize=9)
        fig.suptitle(f"Continuous identity development, protocol {version}: four independent UE sessions",fontsize=12)
        fig.text(.5,.015,"Shading: planned hidden stage. Dashed lines: actual phase setup completion. Missing slots are retained; no flight.",ha="center",fontsize=8)
        fig.tight_layout(rect=[0,.025,1,.915])
        for ext in ("png","pdf"):
            path=output/f"identity_timeline_protocol_{version}.{ext}"
            fig.savefig(path,dpi=180)
            artifacts.append({"path":str(path),"sha256":digest(path)})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    labels=[f"Protocol {i+1}\nN={report['planned_slots_all']}" for i,report in enumerate(reports)]
    timely=[report["captured_slots_all"]-report["late_observer_completions_all"] for report in reports]
    late=[report["late_observer_completions_all"] for report in reports]
    missing=[report["planned_slots_all"]-report["captured_slots_all"] for report in reports]
    x=np.arange(len(labels))
    base=np.zeros(len(labels))
    for values,label,color in ((timely,"Result before assigned slot deadline","#247F79"),
        (late,"Captured; result after deadline","#C39936"),(missing,"No capture","#8996A5")):
        bars=ax.bar(x,values,bottom=base,label=label,color=color,width=.52)
        for bar,value,low in zip(bars,values,base):
            if value:
                center=bar.get_x()+bar.get_width()/2
                if value<5:
                    ax.annotate(str(value),xy=(center,low+value/2),xytext=(center+.35,low+18),
                        fontsize=9,ha="center",arrowprops={"arrowstyle":"-","color":"#66758C"})
                else:
                    ax.text(center,low+value/2,str(value),ha="center",va="center",fontsize=10)
        base+=values
    ax.set_xticks(x,labels)
    ax.set_ylabel("Predeclared slots (denominator frozen per protocol)")
    ax.set_ylim(0,max(report["planned_slots_all"] for report in reports)*1.15)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,fontsize=8,loc="upper right")
    ax.set_title("Ground pilot timing repair; all slots retained")
    fig.text(.5,.015,"Different development sessions; counts describe system timing, not formal method efficacy or independent frames.",ha="center",fontsize=7)
    fig.tight_layout(rect=[0,.035,1,1])
    for ext in ("png","pdf"):
        path=output/f"timing_comparison.{ext}"
        fig.savefig(path,dpi=180)
        artifacts.append({"path":str(path),"sha256":digest(path)})
    plt.close(fig)
    receipt={"status":"RENDERED", "source_sha256":digest(Path(__file__)),
        "sources":[{"path":str(Path(path).resolve()),"sha256":digest(path)} for path in report_paths],
        "artifacts":artifacts,"ai_assistance":"AI assisted plotting; counts and events derived from archived receipts"}
    (output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"RENDERED","artifacts":len(artifacts)},ensure_ascii=False))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",action="append",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.report,args.output)

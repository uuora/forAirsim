"""Export controlled visible-mask balloon labels, keeping one episode together.

This development export has NO train/validation/test partition. Do not train
and evaluate on these same images or randomly split neighboring frames.
"""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from evaluate_t1_ground_perception import digest, save
from t1_ground_perception_probe import visible_mask


def yolo_label(box, width, height):
    values = np.asarray(box,float)
    if values.shape!=(4,) or not np.isfinite(values).all() or not (0<=values[0]<values[2]<=width and 0<=values[1]<values[3]<=height):
        raise ValueError("Invalid half-open visible box")
    x1,y1,x2,y2=values
    return [0,float((x1+x2)/2/width),float((y1+y2)/2/height),float((x2-x1)/width),float((y2-y1)/height)]


def validate_episode_split(rows):
    """Mandatory future split check: whole episode + exact RGB stays together."""
    episodes, images = {}, {}
    counts = {"train":set(),"validation":set(),"test":set()}
    for row in rows:
        split=row["split"]
        if split not in counts:
            raise ValueError("Expected explicit train/validation/test split")
        episode=row["episode_id"]
        sha=row["rgb_sha256"]
        if episode in episodes and episodes[episode]!=split:
            raise ValueError("An episode crosses dataset splits")
        if sha in images and images[sha]!=split:
            raise ValueError("An identical RGB crosses dataset splits")
        episodes[episode]=split;images[sha]=split;counts[split].add(episode)
    if any(not value for value in counts.values()):
        raise ValueError("Independent train, validation and test episodes required")
    return {key:len(value) for key,value in counts.items()}


def run(source_path, output):
    source=json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source["status"]!="PASS" or source["evaluation_segmentation_ids"]!={"T1ProbeTarget":42,"T1ProbeDistractor":43,"T1ProbeOccluder":44}:
        raise ValueError("Verified controlled segmentation IDs required")
    output.mkdir(parents=True,exist_ok=False)
    (output/"images").mkdir();(output/"labels").mkdir()
    report={"status":"RUNNING","kind":"CONTROLLED_VISIBLE_BALLOON_DEVELOPMENT_DATASET",
        "source_report":str(source_path.resolve()),"source_report_sha256":digest(source_path),
        "source_sha256":{Path(__file__).name:digest(__file__)},
        "class_names":{"0":"balloon"},"truth_source":"Controlled visible AirSim instance masks",
        "split":"DEVELOPMENT_UNPARTITIONED","train_val_test_ready":False,
        "independent_UE_sessions":1,"target_and_distractor_share_class":True,"rows":[]}
    for frame in source["frames"]:
        for media in frame["media"].values():
            if digest(media["path"])!=media["sha256"]:
                raise ValueError("Raw media hash mismatch")
        rgb=cv2.imdecode(np.fromfile(frame["media"]["rgb"]["path"],np.uint8),cv2.IMREAD_COLOR)
        seg=cv2.imdecode(np.fromfile(frame["media"]["segmentation"]["path"],np.uint8),cv2.IMREAD_COLOR)
        width,height=frame["image_size_wh"]
        if rgb is None or seg is None or rgb.shape[:2]!=(height,width) or seg.shape!=rgb.shape:
            raise ValueError("Label dimensions differ from RGB")
        labels=[];instances=[]
        for name,segid in (("target",42),("distractor",43)):
            observed=visible_mask(seg,source["segmentation_palette_rgb"][str(segid)])
            if observed!=frame["evaluation_only"][name]:
                raise ValueError("Stored visible-mask labels do not reproduce")
            if observed["bbox_xyxy_pixels"] is not None:
                labels.append(yolo_label(observed["bbox_xyxy_pixels"],width,height))
                instances.append({"evaluation_identity":name,"segmentation_id":segid,**observed})
        image_path=output/"images"/(frame["frame_id"]+".png")
        label_path=output/"labels"/(frame["frame_id"]+".txt")
        shutil.copyfile(frame["media"]["rgb"]["path"],image_path)
        label_path.write_text("".join(" ".join([str(row[0])]+[f"{x:.9f}" for x in row[1:]])+"\n" for row in labels),encoding="utf-8")
        report["rows"].append({"frame_id":frame["frame_id"],"case_id":frame["case_id"],
            "episode_id":frame["session_id"],"split":"DEVELOPMENT_UNPARTITIONED",
            "rgb_sha256":digest(image_path),"image":str(image_path.resolve()),"label":str(label_path.resolve()),
            "label_sha256":digest(label_path),"visible_balloon_count":len(labels),"instances_evaluation_only":instances})
    report.update(status="PASS",frame_count=len(report["rows"]),
                  visible_instance_labels=sum(row["visible_balloon_count"] for row in report["rows"]),
                  negative_frame_count=sum(row["visible_balloon_count"]==0 for row in report["rows"]))
    report["limits"]=["Small controlled development set, not human-verified real-world labels",
        "Visible-mask bounding boxes, not amodal boxes behind occluders",
        "No training executed; no independent validation/test split; cannot claim fine-tuning or held-out accuracy"]
    save(output/"manifest.json",report)
    (output/"README.md").write_text(
        "# 气球开发标注\n\n18 帧来自同一个 UE 会话。类 0 为 balloon，指定目标和干扰气球均属此类。"
        "遮挡无可见气球的帧使用空标签。标签由独立实例分割掩膜复算，未供检测或控制策略使用。\n\n"
        "当前全部保留在 DEVELOPMENT_UNPARTITIONED，未训练，未建验证或测试集。后续应采集独立场景/回合，"
        "按整个 episode 划分，冻结拆分后再训练；validate_episode_split 会拒绝跨组会话或相同图像。\n\n"
        "AI 使用说明：导出与核验代码由 AI 辅助编制，原图、来源、实例标签和哈希见 manifest.json。\n",encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)

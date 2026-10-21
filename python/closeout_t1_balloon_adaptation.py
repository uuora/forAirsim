"""Archive actual detector adaptation results, without promoting T1 efficacy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def run():
    plan_path=ROOT/"configs/t1_balloon_adaptation_v1p1_20261004/plan.json"
    data_path=ROOT/"datasets/t1_balloon_adaptation_v1p1_20261004/manifest.json"
    train_path=ROOT/"logs/t1_balloon_yolo_training_v1p2_20261004/receipt.json"
    eval_path=ROOT/"logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json"
    campaign_path=ROOT/"logs/t1_balloon_adaptation_campaign_v1p1_20261004/campaign_receipt.json"
    plan,data,train,evaluation,campaign=map(load,(plan_path,data_path,train_path,eval_path,campaign_path))
    if any(x["status"]!="PASS" for x in (data,train,evaluation,campaign)):
        raise ValueError("Complete successful collection, export, training and evaluation required")
    if data["plan_sha256"]!=digest(plan_path) or train["dataset_manifest_sha256"]!=digest(data_path):
        raise ValueError("Evidence binding differs")
    if evaluation["training_receipt_sha256"]!=digest(train_path):raise ValueError("Evaluation/training binding differs")
    best=Path(train["best_weights"])
    if digest(best)!=train["best_weights_sha256"]:raise ValueError("Selected checkpoint differs")
    if train["test_images_used"] or train["test_labels_used"]:raise ValueError("Training accessed test")
    if evaluation["unique_test_frames"]!=data["splits"]["test"]["frame_count"] or evaluation["detection_calls"]!=2*evaluation["unique_test_frames"]:
        raise ValueError("Incomplete registered test coverage")
    if any(x["status"]!="RETURNED" for x in evaluation["frames"]):raise ValueError("Detector errors remain")
    with socket.socket() as sock:
        sock.settimeout(.4)
        if sock.connect_ex(("127.0.0.1",41451))==0:raise ValueError("Simulator still listening")
    flights=load(ROOT/"configs/trajectory_timed_pilot_v6p2/center.json")["flight_source_sha256"]
    for name,expected in flights.items():
        if digest(ROOT/"python"/name)!=expected:raise ValueError("Original frozen flight source differs")
    attempts=[]
    for row in plan["episodes"]:
        manifest=load(row["manifest_path"])
        if manifest["status"]!="PASS" or not manifest["engine_closed_after_safe_completion"]:raise ValueError("Owned UE closure missing")
        attempts.append({"episode_id":row["episode_id"],"manifest":row["manifest_path"],
            "manifest_sha256":digest(row["manifest_path"]),"process_id":manifest["engine_process_id"],
            "process_started_utc":manifest["engine_started_utc"],"frames":manifest["saved_frames"],"safe_owned_engine_closed":True})
    failure_path=ROOT/"logs/t1_balloon_adaptation_campaign_v1_20261004/ep01_train/manifest.json"
    failed=load(failure_path)
    if failed["status"]!="FAIL" or not failed["engine_closed_after_safe_completion"]:raise ValueError("Failed first capture closure missing")
    tests_path=ROOT/"logs/t1_balloon_final_tests_20261004.txt"
    tests=tests_path.read_text(encoding="utf-8-sig")
    match=re.search(r"Ran (\d+) tests",tests)
    if not match or "\nOK" not in tests:raise ValueError("Final test log missing PASS")
    output=ROOT/"logs/t1_balloon_adaptation_closeout_v1p2_20261004"
    output.mkdir(exist_ok=False)
    sources=("prepare_t1_balloon_campaign.py","t1_balloon_data_probe.py","test_t1_controlled_mesh.py",
        "build_t1_balloon_dataset.py","test_t1_balloon_dataset.py","specialist_balloon_detector.py",
        "train_t1_balloon_yolo.py","evaluate_t1_balloon_yolo.py","test_t1_specialist_detector.py",Path(__file__).name)
    snapshots=output/"source_snapshot";snapshots.mkdir()
    for name in sources:shutil.copyfile(ROOT/"python"/name,snapshots/name)
    shutil.copyfile(ROOT/"scripts/run-t1-balloon-data-campaign.ps1",snapshots/"run-t1-balloon-data-campaign.ps1")
    receipt={"status":"PASS","kind":"T1_CONTROLLED_DETECTOR_ADAPTATION_CLOSEOUT",
        "completed_utc":datetime.now(timezone.utc).isoformat(),"new_flights":0,"cloud_requests":0,
        "independently_launched_completed_UE_sessions":8,"owned_UE_attempts_including_failed_capture":9,
        "complete_session_receipts":attempts,"failed_capture_manifest":str(failure_path),
        "failed_capture_manifest_sha256":digest(failure_path),"failed_capture_frames_retained_not_used":failed["saved_frames"],
        "failed_training_receipt":str(ROOT/"logs/t1_balloon_yolo_training_v1p1_20261004/receipt.json"),
        "failed_training_receipt_sha256":digest(ROOT/"logs/t1_balloon_yolo_training_v1p1_20261004/receipt.json"),
        "failed_training_reason":"Polars CPU-feature parsing during checkpoint CSV metadata reading; no test access",
        "retry_scope":"stdlib CSV metadata adapter; original frozen seed/data/20-epoch protocol retained",
        "bound_artifacts":{key:{"path":str(path),"sha256":digest(path)} for key,path in
            (("plan",plan_path),("dataset",data_path),("training",train_path),("evaluation",eval_path),("campaign",campaign_path),("test_log",tests_path))},
        "frame_count":data["frame_count"],"visible_instance_labels":data["visible_instance_labels"],
        "negative_frame_count":data["negative_frame_count"],"splits":data["splits"],
        "specialist_weights":str(best),"specialist_weights_sha256":digest(best),
        "completed_training_epochs":train["completed_epochs"],"summary_by_model":evaluation["summary_by_model"],
        "specialist_test_map":evaluation["specialist_map"]["results"],
        "tests_passed":int(match.group(1)),"original_frozen_flight_source_sha256":flights,
        "source_sha256":{name:digest(snapshots/name) for name in (*sources,"run-t1-balloon-data-campaign.ps1")},
        "persistent_identity_tracking_complete":False,"new_detector_flight_integration_complete":False,
        "formal_T1_method_effect_complete":False,"rpc_listener_after":False,
        "limits":plan["limits"],"ai_assistance":"AI assisted implementation, archive and evidence-bound Chinese summary"}
    save(output/"receipt.json",receipt)
    detector_config={"status":"CONTROLLED_DETECTOR_PILOT_TESTED_INTEGRATION_PENDING",
        "class_names":{"0":"balloon"},"backend":"specialist_balloon_detector.SpecialistBalloonDetector",
        "weights":str(best),"weights_sha256":digest(best),"training_receipt":str(train_path),
        "training_receipt_sha256":digest(train_path),"imgsz":640,"confidence":.25,"nms_iou":.5,"device":"cpu",
        "vlx_adapter_injection":"YoloVlxBackend(detector=SpecialistBalloonDetector(...), vlx_client=...)",
        "designated_identity_established_by_detection":False,"flight_integration_verified":False}
    save(ROOT/"configs/t1_balloon_detector_v1p2.json",detector_config)
    before_path=ROOT/"logs/t1_balloon_task_registry_before_20261004.json"
    registry=load(ROOT/"configs/thesis_simulation_tasks.json");before=load(before_path)
    if registry["tasks"][1:]!=before["tasks"][1:]:raise ValueError("T2/T3 changed unexpectedly")
    t1=registry["tasks"][0]
    registry["active_stage"]="task_1_persistent_identity_recovery_and_specialist_detector_trajectory_integration"
    t1["next_gate"]="continuous_sequence_designated_identity_tracking_recovery_then_new_trajectory_execution"
    t1["perception_integration_20261004"]["specialist_YOLO_training_completed"]=True
    t1["detector_adaptation_20261004"]={"status":"CONTROLLED_EPISODE_GROUPED_TRAINING_AND_TEST_COMPLETED",
        "receipt":str((output/"receipt.json").relative_to(ROOT)).replace("\\","/"),
        "new_flights":0,"cloud_requests":0,"frame_count":data["frame_count"],
        "train_validation_test_sessions":[4,2,2],"training_epochs":train["completed_epochs"],
        "test_frames":evaluation["unique_test_frames"],"test_episodes":evaluation["independent_test_episodes"],
        "specialist_weights_sha256":digest(best),"formal_method_effect_complete":False}
    save(ROOT/"configs/thesis_simulation_tasks.json",registry)
    a=evaluation["summary_by_model"]["original_COCO_sports_ball"];b=evaluation["summary_by_model"]["specialist_balloon"]
    def percent(x):return "未定义" if x is None else f"{x*100:.1f}%"
    doc=ROOT/"docs/research-plan-20261004/任务一专用YOLO首轮结果与后续闭环.md"
    text=f"""# 任务一专用 YOLO 首轮结果与后续闭环

更新：2026-10-04。已完成按 UE 会话分组的首轮气球检测采集、训练和测试；任务一整体仍待持续身份与轨迹实验。[本轮归档](../../logs/t1_balloon_adaptation_closeout_v1p2_20261004/receipt.json)

## 数据与训练

采集前固定八个 UE 会话的种子和分组，改变球体尺度、相机距离/位置/姿态、目标位置及双球间距。每会话包含十二单球、八双球、四全遮挡负例，各一帧；共 {data['frame_count']} 张图、{data['visible_instance_labels']} 个可见框、{data['negative_frame_count']} 张负例。所有实例都标为 class 0 balloon，身份标签不输入模型。[冻结计划](../../configs/t1_balloon_adaptation_v1p1_20261004/plan.json)、[数据清单](../../datasets/t1_balloon_adaptation_v1p1_20261004/manifest.json)

| 分组 | 独立 UE 会话 | 图像 | 可见气球框 | 负例 |
|---|---:|---:|---:|---:|
| 训练 | 4 | 96 | 112 | 16 |
| 验证 | 2 | 48 | 56 | 8 |
| 测试 | 2 | 48 | 56 | 8 |

整会话分组，相同 PNG 和完全相同的解码 RGB 不跨组，本批组内重复为零。近似画面和共同地图/材质仍存在，两个测试会话才是本批独立来源数量，四十八帧是会话内观测。[来源与重复核验](../../datasets/t1_balloon_adaptation_v1p1_20261004/manifest.json)

在 RTX 3070 Laptop GPU 上按预注册参数从原 YOLO11n 微调 {train['completed_epochs']} 个 epoch，验证集选 best.pt。训练仅使用九十六训练图和四十八验证图，测试图片/标签使用计数为零；不使用测试结果改阈值或再次选模。[训练回执](../../logs/t1_balloon_yolo_training_v1p2_20261004/receipt.json)

## 固定阈值的独立来源测试

同 CPU、640 输入、置信度阈值 0.25、NMS IoU 0.5；框以 IoU≥0.5 最大一对一匹配，重复框计误检。每方法四十八帧全部返回，检测耗时用预热后的 perf_counter_ns 测量，未含 AirSim 采集和 VLX 请求。[逐帧与分会话结果](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json)

| 指标 | 原 COCO sports ball 映射 | 专用 balloon |
|---|---:|---:|
| 正确检出实例 / 56 | {a['true_positive']}/56 | {b['true_positive']}/56 |
| 漏检实例 | {a['false_negative']} | {b['false_negative']} |
| 误检框 | {a['false_positive']} | {b['false_positive']} |
| 检测精确率 | {percent(a['precision'])} | {percent(b['precision'])} |
| 实例召回率 | {percent(a['recall'])} | {percent(b['recall'])} |
| 双球均检出 / 16 帧 | {a['both_balloons_detected_frames']}/16 | {b['both_balloons_detected_frames']}/16 |
| 有误检的负例 / 8 帧 | {a['negative_frames_with_false_positive']}/8 | {b['negative_frames_with_false_positive']}/8 |
| 检测中位耗时 | {a['latency_ms_median']:.2f} ms | {b['latency_ms_median']:.2f} ms |

专用模型另有低阈值 mAP 扫描，记录见 [mAP 原始结果](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/specialist_map.json)。它与上表固定阈值的精确率/召回率不是同一计算口径；原 COCO 八十类未直接对单类标签计算 mAP。

## 失败保留与代码接入

首版采集在第十九场景出现重建物体名称冲突，十八帧部分记录保留并排除。新入口直接更新已创建物体，全八会话均安全关闭。训练首试在第一 epoch 保存权重时遇到 Polars CPU 特征解析错误；重试仅使用标准 CSV 读取保存元数据，训练参数和数据保持冻结，失败快照均保留。[失败采集](../../logs/t1_balloon_adaptation_campaign_v1_20261004/ep01_train/manifest.json)、[失败训练](../../logs/t1_balloon_yolo_training_v1p1_20261004/receipt.json)、[重试适配说明](../../logs/t1_balloon_yolo_training_v1p2_20261004/receipt.json)

专用模型必须用 SpecialistBalloonDetector，严格检查模型名称为 {{0: balloon}}；旧 COCO 映射不会识别专用头。新检测器的候选格式已兼容 YoloVlxBackend(detector=...) 注入，VLX 继续负责低频语义复核。此处完成类别检测，尚未验证持续指定实例身份。[检测器](../../python/specialist_balloon_detector.py)、[接入配置](../../configs/t1_balloon_detector_v1p2.json)

本轮 {receipt['tests_passed']} 项相关测试通过，原十二冻结飞行模块不变，新增飞行和云请求均为零。[测试日志](../../logs/t1_balloon_final_tests_20261004.txt)、[归档源码](../../logs/t1_balloon_adaptation_closeout_v1p2_20261004/receipt.json)

## 论文加入什么，下一步做什么

第三章感知实现与实验设置应加入：分组/种子/数据统计、可见掩膜监督、同类干扰气球、训练与验证选模、上述原模型对照、阴性误检、双球漏检、处理成本和失败案例。训练数据属于受控 AirSim 数据，不足以声称跨地图/外观泛化；这些检测结果不证明轨迹优化收益。[第三章内容与完成标准](任务一论文内容与完成标准.md)

接下来按顺序推进：

1. 接入专用候选，采集持续单目标序列，检查相似干扰、遮挡、重现、语义延迟；实现身份状态、唯一关联和恢复。
2. 将有效身份观测与表面估计接入新版轨迹闭环，完成运动范围和执行时效准入。
3. 固定跟随、距离、可见性、身份联合四基线使用相同检测后端；颜色检测/专用 YOLO 与 VLX 开关作为独立消融。
4. 冻结正式回合与全部失败分母，采集独立飞行重复后写第三章方法效果结果。

方法论文的主要证据仍是身份保持和轨迹优化实验。YOLO 单类检测分数不是校准身份概率；测试集已消费，后续改模型需另建未参与调参的测试版本。[正式协议草案](../../configs/t1_completion_protocol_v1.json)

实现参考：[Ultralytics 训练文档](https://docs.ultralytics.com/modes/train)、[验证文档](https://docs.ultralytics.com/modes/val)。AI 使用说明：代码、参数组织和结果说明由 AI 辅助，数值来自上述实际仿真/训练/测试回执，不代替导师和学校评审。
"""
    doc.write_text(text,encoding="utf-8")
    save(output/"document_receipt.json",{"path":str(doc),"sha256":digest(doc),"task_registry":str(ROOT/"configs/thesis_simulation_tasks.json"),
        "task_registry_sha256":digest(ROOT/"configs/thesis_simulation_tasks.json"),"T2_T3_unchanged":True})
    print(json.dumps({"status":"PASS","document":str(doc),"summary":evaluation["summary_by_model"]},ensure_ascii=False))


if __name__=="__main__":run()

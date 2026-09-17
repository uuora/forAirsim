# YOLO 目标识别基线

## 当前实现

2026-09-16 已接入官方预训练 `yolo11n.pt`，使用 Ultralytics 8.4.153、PyTorch 2.11.0+cu128 和 RTX 3070 Laptop GPU。权重 SHA-256 为 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。Ultralytics 包元数据标注许可证为 AGPL-3.0；后续分发或产品使用必须单独复核许可证适用性。

当前没有项目专用训练。应用层只作两个临时候选映射：COCO `boat → vessel`、`sports ball → balloon`。这用于验证数据接口，不表示两个概念在真实海事场景中完全等价。

## 已保存图像基线

双击项目根目录的 `Run-YOLO-Baseline.cmd`，或执行：

```powershell
.venv\python.exe python\run_yolo_baseline.py
```

正式报告：[report.json](../logs/yolo_baseline_20260916_162112_485440/report.json)。机器可读预测：[predictions.jsonl](../logs/yolo_baseline_20260916_162112_485440/predictions.jsonl)。临时类别映射总览：[demo_mapped_025_contact_sheet.jpg](../logs/yolo_baseline_20260916_162112_485440/demo_mapped_025_contact_sheet.jpg)。

10 张空中图共得到 34 个置信度不低于 0.10 的原始 COCO 预测，其中 31 个映射为项目候选类别。阈值描述如下：

|置信度阈值|3 张干净种子图 TP/FP/FN|全部 10 图候选预测|未映射预测|
|---:|---:|---:|---:|
|0.10|9 / 0 / 0|31|3|
|0.25|9 / 0 / 0|27|0|
|0.50|7 / 0 / 2|23|0|

0.25 只作为当前保存图演示阈值：它在这 3 张图上保留全部 9 个匹配，同时去掉低置信度 `bird/frisbee/person`。这不是经过独立验证的工作点。样本来自一个场景和一个采集会话，`9/9` 只能说明冒烟链路吻合，不能作为准确率或泛化性能声明。

GPU 平均纯推理约 18.8 毫秒/张；包含 Python、数据组装和顺序批处理的主机墙钟约 65.3 毫秒/张。该数据不包含 AirSim 相机 RPC 时间，不能直接当成闭环感知时延。

## 修正构图会话推理

`run_yolo_baseline.py` 现在可从采集报告动态读取航点，并通过 `--no-evaluation` 对尚未建立审核真值的会话只做推理。修正构图会话报告为 [report.json](../logs/yolo_baseline_20260916_164315_165553/report.json)，预览为 [demo_mapped_025_contact_sheet.jpg](../logs/yolo_baseline_20260916_164315_165553/demo_mapped_025_contact_sheet.jpg)。

10 张图在 0.10 阈值下得到 40 个项目候选，对应每张图一个 vessel 和三个 balloon 候选。在 0.25 展示阈值下保留 39 个；`rear_left / DroneB` 的近红气球候选置信度为 0.240。该会话没有独立真值评分，因此这些数字只描述候选覆盖，不能写成召回率或准确率，也说明 0.25 尚不能固定为任务工作阈值。

低太阳会话的只推理报告为 [report.json](../logs/yolo_baseline_20260916_170330_583701/report.json)，预览为 [demo_mapped_025_contact_sheet.jpg](../logs/yolo_baseline_20260916_170330_583701/demo_mapped_025_contact_sheet.jpg)。候选计数为：0.10 阈值 40 个、0.25 阈值 39 个、0.50 阈值 37 个。`rear_center / DroneB` 的近红气球在 0.10 输出中的置信度为 0.202，因此 0.25 展示阈值会漏掉它。这些仍是候选覆盖统计，不是准确率或召回率。

## 实时 AirSim 快照

双击 `Run-YOLO-Live-Snapshot.cmd`。脚本只读取两机 `front_center`，不取得 API 控制、不解锁、不起飞。正式报告：[predictions.json](../logs/yolo_live_20260916_162355_161007/predictions.json)。双机预览：[live_yolo_contact_sheet.jpg](../logs/yolo_live_20260916_162355_161007/live_yolo_contact_sheet.jpg)。

停机位实跑中，两机均只检测到黄色气球，映射置信度为 0.80 和 0.84；红气球和船受甲板遮挡，没有达到 0.25。GPU 推理为 23.8–29.5 毫秒/张，相机 RPC 为 90–165 毫秒/张。该结果支持继续使用空中观察位，但两帧没有同步真值，不能据此计算实时准确率。

## 研究边界与下一步

- 当前是官方 COCO 预训练模型，不是本项目训练模型；
- 3 张干净图不足以训练或验证，不能随机拆分同一会话制造验证集；
- 下一轮需在空中视点采集独立的 baseline、low_sun、raised_waves、occlusion 会话；
- 数据集按采集会话分组后，才能比较预训练基线、颜色方法和项目微调模型；
- YOLO 输出保持在独立 JSON 中，不能覆盖原图、替代碰撞命中裁判或直接驱动飞行控制。

本文档和框复核由 AI 辅助完成，尚无独立人工标注认证。

# VLX-Seek 接入预检与海事图像回放（2026-09-26）

**最新进展：** 三个独立进程会话的 36 帧左右布局对照、AI 辅助可见框复核、公开 VLX 演示实跑失败记录及后续批量运行入口，见 `docs/海事左右布局探索性对照与VLX实跑状态.md`。下文的 30 帧表格是此前六条件回放的历史结果。

## 目标与实际完成范围

在既有 `MaritimeHarborV2` 单机悬停采集上，使用同一批 30 张 RGB 图像接通 `YOLO → 区域候选框 → VLX-Seek grounding_single → 实例评价` 的离线接口。VLX 输入只有图像、候选框和固定文本描述；保存报告里的 AirSim 投影框仅用于评价，不进入 YOLO、红色区域提案或 VLX。此阶段不向 AirSim 发送飞行命令。

实现为 `python/vlx_seek_probe.py`。VLX 调用对应官方 `VLXSeekWorker.ground(image, bbox_list, description, lang="en", temperature=0.0)`；候选框保持原图像素坐标，最多 100 个。返回框必须属于输入候选框，否则脚本拒绝评分。默认描述固定为 `the rightmost red balloon in the image`，实验前确定，不根据每帧的 AirSim 框变化。

后续的左右位置预检把每组的任务描述预先写入采集配置。回放入口会优先读取报告中逐帧保存的描述；只有旧报告未提供该字段时才使用上述固定默认值。位置预检结果见 `docs/单机海事地图目标判别实验.md`。

官方目前只发布 [VLX-Seek 1.5-10B 的推理代码和权重](https://github.com/om-ai-lab/VLX-Seek/blob/main/README.md)，[模型权重文件约 20 GB](https://huggingface.co/omlab/VLX-Seek-1.5-10B/blob/main/model.safetensors)。官方[推理指南](https://github.com/om-ai-lab/VLX-Seek/blob/main/docs/inference.md)允许外部检测器提供 `[x1,y1,x2,y2]` 候选框，因而可直接接已有 YOLO。当前 Windows 主机只有 8 GB GPU 显存、约 15.3 GiB 物理内存，未下载 10B 权重、未完成 VLX 实际推理；本报告中的 VLX 准确性和时延均为 **未测**。权重所需内存还要包含推理激活和视觉模块，20 GB 文件大小本身就超过本机显存。没有把接口测试冒充模型推理。

## 本机已执行的回放

采集源：`logs/maritime_single_20260925_205856_936729/report.json`，六条件各 5 帧。YOLO 用已有 `models/yolo11n.pt`、置信度 0.10、`imgsz=1280`、CUDA 设备 0。报告：

- `logs/vlx_seek_probe_yolo_20260926.json`：仅 YOLO 候选；
- `logs/vlx_seek_probe_yolo_color_20260926.json`：YOLO + 原有严格红气球轮廓候选；
- `logs/vlx_seek_probe_red_20260926.json`：YOLO + 宽松可见红色连通域候选。

| 条件 | 帧数 | YOLO 目标候选 IoU≥0.5 | YOLO 最高置信气球选中目标 | YOLO 最右气球规则选中目标 | YOLO+宽松红区目标候选 IoU≥0.1 |
|---|---:|---:|---:|---:|---:|
| baseline | 5 | 5 | 4 | 5 | 5 |
| target_shift | 5 | 5 | 5 | 5 | 5 |
| similar_red | 5 | 5 | 1 | 5 | 5 |
| target_occluded | 5 | 0 | 0 | 0 | 5 |
| low_sun | 5 | 5 | 2 | 5 | 5 |
| raised_waves | 5 | 5 | 4 | 5 | 5 |

宽松红区在遮挡条件的 5 帧均产生与目标投影框 IoU≥0.1 的区域，但 **0/5 达到 IoU≥0.5**。这些连通域可能混入船体的橙红像素，需要人工复核可见区域。严格红轮廓模式对遮挡没有增加目标候选。`similar_red` 帧中 YOLO 能分别提出两个红气球候选，最高置信度选择在 5 帧中只选中指定目标 1 次；简单“最右气球”规则却是 5 次。当前地图的固定空间关系使这条规则很强，不能把后续 VLX 按同一提示的表现直接解释为语义推理收益。

以上 5 帧是同一会话中的连续图像；AirSim 投影框包含遮挡部分且图像和框由相邻 RPC 获得。表中是这批帧的回放计数，不是独立样本准确率或真实可见区域召回率。目标身份 `balloon_001` 只在评价阶段使用。

## 重放命令

在 `D:\forAirsim`：

```powershell
& .\.venv\python.exe .\python\vlx_seek_probe.py `
  --report .\logs\maritime_single_20260925_205856_936729\report.json `
  --output .\logs\vlx_seek_probe_red_repeat.json `
  --proposal-mode yolo_red_regions
```

在具备足够显存的机器上，先按[官方仓库](https://github.com/om-ai-lab/VLX-Seek)的说明配置独立 Python 环境，并准备完整的本地 VLX-Seek 源码与 10B 权重。不要在当前 YOLO 环境中直接覆盖其 PyTorch/Transformers 版本。把原采集报告及其 30 张图片、上一步生成的候选报告复制到该机器，保留报告和图片的同目录关系。第二步读取缓存候选，不再加载 YOLO 或原权重：

```powershell
& .\<vlx-env>\python.exe .\python\vlx_seek_probe.py `
  --report .\logs\maritime_single_20260925_205856_936729\report.json `
  --output .\logs\vlx_seek_probe_actual.json `
  --proposals-report .\logs\vlx_seek_probe_red_20260926.json `
  --proposal-mode yolo_red_regions `
  --run-vlx --vlx-repo .\downloads\VLX-Seek `
  --model-path .\models\VLX-Seek-1.5-10B
```

建议先用 `--proposal-mode yolo` 单独测双红气球的实例选择，再用 `yolo_red_regions` 测遮挡候选；对照同一帧的最高置信 YOLO 和最右规则。首次真实 VLX 运行需要记录模型版本与设备信息，并人工复核遮挡区域。接下来应增加目标与干扰物左右位置交替、视角变化及独立采集会话，使简单空间规则不再恒定正确，然后才能判断 VLX 是否增加有效信息。

文档和代码由 AI 辅助生成；图像可见区域尚无独立人工标注认证。

# 海事左右布局探索性对照与 VLX 实跑状态（2026-09-26）

## 数据和复核来源

`logs/maritime_layout_collection_20260926/collection_manifest.json` 索引三个独立重启的 Unreal 进程、每次四种布局各三帧，共 36 张 1280×720 图像。四种布局让指定目标 `balloon_001` 分别位于两只红气球的左侧或右侧，近距与远距各一对。三个采集报告均为 PASS，落地与场景恢复已确认。

我查看了每个会话 12 张图像的放大裁剪和候选框。36/36 张均显示两只红色气球；绿色目标框与青色干扰框的左右关系符合采集前写入的任务。候选框由原有红色轮廓检测器生成，随后经过 **Codex AI 视觉复核**，保存在 `review_queue.csv`，状态 `AI_REVIEWED`。这不是独立人工标注，也不是完整球体的像素掩膜；框主要覆盖可见红色球体。三个放大复核图为 `session_{1,2,3}_ai_review_draft.jpg`。AirSim 投影框只作场景身份参考，没有复制为可见框。AI 复核队列另存为 `review_queue_ai_snapshot.csv`；人类复核工具可在原图上从空框重新绘制，覆盖主队列中的 AI 状态。

`python/validate_maritime_layout_review.py` 默认仍只把人工 `VERIFIED` 视为正式评分就绪；添加 `--allow-ai-reviewed` 才返回 `READY_FOR_EXPLORATORY_EVALUATION`。当前结果：人工核验 0、AI 复核 36、待复核 0。

## 三会话对照

使用同一批图像、已有 `models/yolo11n.pt`（COCO `sports ball` 映射为项目 `balloon`）和预注册的左/右文本任务。YOLO 置信度阈值 0.10、`imgsz=1280`，保存候选报告 `session_{1,2,3}_yolo_proposals.json`。比较脚本是 `python/compare_maritime_layout_baselines.py`，逐帧结果和聚合在 `exploratory_baseline_comparison.json`。表中数字是选中指定目标的帧数，每会话 12 帧：

| 方法 | 会话 1 | 会话 2 | 会话 3 | 合计 |
|---|---:|---:|---:|---:|
| YOLO 最高置信气球 | 4 | 5 | 5 | 14/36 |
| YOLO 全部气球中按提示选左/右 | 6 | 6 | 6 | 18/36 |
| YOLO 候选经红色像素过滤后按提示选左/右 | 12 | 12 | 12 | 36/36 |
| 严格颜色轮廓，选最大面积 | 10 | 6 | 6 | 22/36 |
| 严格颜色轮廓，按提示选左/右 | 12 | 12 | 12 | 36/36 |
| 宽松红色连通域，按提示选左/右 | 6 | 6 | 6 | 18/36 |

YOLO 对指定目标和另一只红气球的候选覆盖，在三个会话中均为 12/12 帧达到与 AI 辅助可见框 IoU≥0.5。图中另有黄色气球，因此“不先识别红色、只在所有气球中选最左”会选错。宽松红色连通域包含其他红色场景区域，直接取最左/右也不可靠。

**解释边界：** AI 可见框的建议来自严格颜色检测器，所以颜色法的精确定位表现不能据此独立验证。此处只报告实例选择的探索性计数；三会话使用相同场景参数，连续帧高度相关，不报告置信区间或泛化准确率。提示词直接给出了左/右，几何与颜色规则已在本批数据得到 36/36；即使将来 VLX 得到 36/36，也不能仅凭本批数据证明语义模型带来增益。后续需要预注册不由单一左右规则解出的属性或遮挡任务，并在不同视角、光照及布景中独立采集。

## VLX 实际推理状态

`python/vlx_seek_probe.py` 已按[官方推理指南](https://github.com/om-ai-lab/VLX-Seek/blob/main/docs/inference.md)接入 `VLXSeekWorker.ground(image, bbox_list, description)`；本批三份 YOLO 候选报告可被独立 VLX 环境直接复用。官方文档确认可使用外部检测器提供原图像素框，最多 100 个候选。上述候选报告的 `vlx_executed` 均为 `false`，不能解释为 VLX 结果。

截至本次检查，[官方仓库](https://github.com/om-ai-lab/VLX-Seek)公开的是 1.5-10B 推理代码与权重；[模型文件](https://huggingface.co/omlab/VLX-Seek-1.5-10B/blob/main/model.safetensors)约 20 GB。本机 RTX 3070 Laptop GPU 为 8 GB 显存，当前没有完整权重，未在本机执行 10B 推理。

额外使用公开的[社区演示 Space](https://huggingface.co/spaces/hugging-apps/vlx-seek-1-5-10b-demo)上传一张 AirSim 图，分别调用 `grounding_single` 和 `vqa`。两次均收到不含内部细节的 Gradio 服务端异常，结果记录为 `vlx_hosted_smoke_session_1.json` 和 `vlx_hosted_vqa_session_1.json`；没有模型输出。该 Space 的公开 API 不接收外部 `bbox_list`，而使用自己的 WeDetect，因此即便恢复也只能做 VLX 单帧冒烟测试，不能验证 YOLO→VLX 链路。

**本轮没有取得 VLX 的准确率、时延或是否胜过基线的证据。** 后续实跑需要能加载官方 10B 的计算环境和完整源码/权重。将三份报告、各自图像、三份候选报告、`collection_manifest.json` 与 `python/vlx_seek_probe.py`、`python/run_vlx_layout_batch.py` 复制到该环境。批量入口只加载一次模型，按会话分别写出原始回答、返回框和耗时；在真实运行前后应另记官方源码版本、权重版本与设备。示例：

```powershell
& .\<vlx-env>\python.exe .\python\run_vlx_layout_batch.py `
  --collection .\logs\maritime_layout_collection_20260926 `
  --logs-dir .\logs --vlx-repo .\downloads\VLX-Seek `
  --model-path .\models\VLX-Seek-1.5-10B --vlx-device cuda `
  --output-dir .\logs\vlx_layout_actual
```

## 本机复现命令

```powershell
& .\.venv\python.exe .\python\validate_maritime_layout_review.py `
  .\logs\maritime_layout_collection_20260926\collection_manifest.json --allow-ai-reviewed

& .\.venv\python.exe .\python\compare_maritime_layout_baselines.py `
  --manifest .\logs\maritime_layout_collection_20260926\collection_manifest.json `
  --proposals .\logs\maritime_layout_collection_20260926\session_1_yolo_proposals.json `
              .\logs\maritime_layout_collection_20260926\session_2_yolo_proposals.json `
              .\logs\maritime_layout_collection_20260926\session_3_yolo_proposals.json `
  --output .\logs\maritime_layout_collection_20260926\exploratory_baseline_comparison.json
```

图像复核、比较脚本和文档由 AI 辅助完成；本轮没有人类独立标注或 VLX 有效推理结果。

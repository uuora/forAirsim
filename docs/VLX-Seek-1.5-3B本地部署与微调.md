# VLX-Seek-1.5-3B 本地部署与微调指南

## 一、能否本地部署？

### 硬件需求评估

根据 Om AI Lab 发布的信息，VLX-Seek-1.5-3B 的硬件需求：

| 项目 | 要求 | 你的配置 | 可行性 |
|------|------|---------|--------|
| **模型参数** | 3B (30亿) | - | ✓ |
| **权重文件** | 约 6-8 GB | - | ✓ |
| **推理显存** | 约 8-12 GB | **8 GB** | ⚠️ **临界** |
| **系统内存** | 16 GB+ | 15.3 GB | ⚠️ **临界** |

### 结论

**理论上可以部署，但会很吃紧：**

1. **纯推理模式**：勉强可行
   - 使用 `bfloat16` 或 `int8` 量化
   - 关闭其他 GPU 程序
   - 批量大小设为 1
   - 预计实际显存占用：6-10 GB

2. **微调模式**：**显存不足**
   - 微调需要存储梯度和优化器状态
   - 通常需要 3-4× 推理显存
   - 8 GB 显存无法完成标准微调

---

## 二、本地部署方案

### 方案 A：官方推理代码（推荐）

根据 VLX-Seek GitHub 仓库：

```bash
# 1. 克隆官方仓库
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 3B 模型权重
# 从 HuggingFace 下载（需要约 6-8 GB 空间）
# https://huggingface.co/omlab/VLX-Seek-1.5-3B
```

**推理示例**：

```python
from vlx_seek_worker import VLXSeekWorker
from PIL import Image

# 初始化（使用量化以节省显存）
worker = VLXSeekWorker(
    model_path="./VLX-Seek-1.5-3B",
    device="cuda",
    load_in_8bit=True  # 8-bit 量化，节省约 40% 显存
)

# 推理
image = Image.open("test.jpg").convert("RGB")
boxes = [[50, 100, 150, 200], [250, 100, 350, 200]]
result = worker.ground(
    image=image,
    bbox_list=boxes,
    description="the rightmost red balloon",
    temperature=0.0
)

print(result["result_bbox_list"])
```

### 方案 B：使用 vLLM 推理引擎

vLLM 针对 LLM 推理优化，显存占用更低：

```bash
pip install vllm

# 启动推理服务
vllm serve omlab/VLX-Seek-1.5-3B \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --dtype bfloat16
```

### 方案 C：云端 API（当前使用）

如果本地部署困难，继续使用 Om AI Lab 的云端 API：

- ✓ 无显存限制
- ✓ 无需下载模型
- ✓ 稳定性更高
- ✗ 需要网络
- ✗ 按 token 计费

---

## 三、微调方案

### 能否用 LLamaFactory 微调？

**理论上可以，但有重要限制：**

#### 1. LLamaFactory 兼容性

VLX-Seek-1.5-3B 基于 **Qwen2-VL** 架构，LLamaFactory **已支持** Qwen2-VL 系列：

```bash
# 检查 LLamaFactory 支持的模型
llamafactory-cli list

# 应该能看到 qwen2_vl 在列表中
```

#### 2. 显存限制

**你的 8 GB 显存无法直接微调 3B 模型**。需要使用以下技术：

##### 方案 1：LoRA 微调（推荐）

只训练少量参数，显著降低显存需求：

```bash
llamafactory-cli train \
  --model_name_or_path ./VLX-Seek-1.5-3B \
  --stage sft \
  --do_train \
  --dataset custom_grounding \
  --template qwen2_vl \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./output \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --fp16 \
  --save_steps 100
```

**显存估算**：

- 基础模型：6 GB (frozen)
- LoRA 参数：~200 MB
- 梯度/优化器：~1 GB
- 激活值：~2 GB
- **总计：约 9-10 GB** ⚠️ **超出 8 GB**

##### 方案 2：QLoRA（量化 + LoRA）

进一步降低显存：

```bash
llamafactory-cli train \
  --quantization_bit 4 \
  --finetuning_type lora \
  --lora_rank 4 \
  ...
```

**显存估算**：

- 量化模型：3-4 GB
- LoRA 参数：~100 MB
- 梯度/优化器：~500 MB
- 激活值：~1.5 GB
- **总计：约 5-6 GB** ✓ **可行**

##### 方案 3：云端微调

如果本地仍然不足，考虑：

1. **Google Colab Pro**（~$10/月）
   - Tesla T4（16 GB 显存）
   - 可以完成 LoRA 微调

2. **AutoDL / 恒源云**（按小时计费）
   - RTX 3090（24 GB）：~2元/小时
   - 适合短期微调项目

3. **Om AI Lab 平台**
   - 可能提供微调服务（需询问）

---

## 四、微调数据准备

### 数据格式

LLamaFactory 需要 JSON 格式的数据集：

```json
[
  {
    "instruction": "识别每个：最右边的红色气球。",
    "input": "image_path_or_url",
    "output": "<ground>最右边的红色气球</ground><objects><obj2></objects>",
    "bbox_proposals": [[50, 100, 150, 200], [250, 100, 350, 200], [450, 100, 550, 200]],
    "ground_truth_bbox": [450, 100, 550, 200]
  }
]
```

### 从 AirSim 数据转换

创建 `python/prepare_vlx_finetuning_data.py`：

```python
"""Convert maritime experiment data to VLX-Seek fine-tuning format."""
import json
from pathlib import Path

def convert_to_training_data(report_path, output_path):
    """Convert maritime report to LLamaFactory format.
    
    Args:
        report_path: Path to maritime_single report.json
        output_path: Output JSON path
    """
    report = json.loads(Path(report_path).read_text())
    
    training_data = []
    for condition in report["conditions"]:
        for frame in condition["frames"]:
            # 构造训练样本
            sample = {
                "instruction": frame.get("task_description", 
                    "识别每个：最右边的红色气球。"),
                "input": str(Path(report_path).parent / frame["frame"]),
                "output": f"<ground>目标气球</ground><objects><obj0></objects>",
                "bbox_proposals": [
                    list(box) for box in frame.get("proposals", [])
                ],
                "ground_truth_bbox": frame["simulator_boxes_xyxy"][
                    report["config"]["target_id"]
                ]
            }
            training_data.append(sample)
    
    Path(output_path).write_text(
        json.dumps(training_data, ensure_ascii=False, indent=2)
    )
    print(f"Saved {len(training_data)} training samples to {output_path}")

if __name__ == "__main__":
    convert_to_training_data(
        "logs/maritime_single_20260925_205856_936729/report.json",
        "data/vlx_maritime_train.json"
    )
```

### 数据增强建议

由于当前只有 30 帧数据，建议：

1. **收集更多场景**
   - 不同时间（晨、午、暮）
   - 不同天气（晴、阴、雾）
   - 不同距离（近、中、远）

2. **数据增强**
   - 亮度调整
   - 对比度变化
   - 轻微旋转/翻转

3. **最小数据量**
   - LoRA 微调：建议 500+ 样本
   - 全量微调：建议 5000+ 样本

---

## 五、实际操作建议

### 立即可行：

1. **继续使用云端 API**（当前方案）
   - 成本可控
   - 稳定可靠
   - 专注于实验设计

### 中期计划：

2. **本地部署推理**（如果需要离线运行）
   - 下载 VLX-Seek-1.5-3B 权重
   - 使用 8-bit 量化
   - 仅用于推理，不微调

### 长期规划：

3. **如果需要微调**
   - 收集 500+ 标注样本
   - 使用云端 GPU（Colab Pro / AutoDL）
   - QLoRA 方式微调
   - 导出 LoRA 权重后本地推理

---

## 六、成本对比

| 方案 | 初期投入 | 运行成本 | 适用场景 |
|------|---------|---------|---------|
| **云端 API** | 0 | ~0.01元/帧 | 当前验证阶段 |
| **本地推理** | 权重下载时间 | 电费 | 大量离线推理 |
| **云端微调** | 0 | ~20-50元/次 | 模型定制化 |
| **本地微调** | 硬件升级 | - | **不推荐** |

---

## 七、快速决策表

### 你应该选择哪种方案？

```
需要离线运行？
├─ 否 → 继续使用云端 API ✓
└─ 是 → 本地部署推理
         └─ 需要微调？
             ├─ 否 → 下载 3B 权重 + 8-bit 量化推理 ✓
             └─ 是 → 云端微调（AutoDL/Colab）✓
                     └─ 微调完成后导出 LoRA 权重
                         └─ 本地加载 (基础模型 + LoRA)
```

---

## 八、下一步行动

### 如果你想尝试本地部署：

1. **测试显存占用**：
   ```bash
   # 安装官方代码
   git clone https://github.com/om-ai-lab/VLX-Seek.git
   cd VLX-Seek
   pip install -r requirements.txt
   
   # 下载 3B 权重（而非 10B）
   # 尝试加载并监控显存
   nvidia-smi -l 1  # 实时监控
   ```

2. **如果显存不足**：
   - 尝试 `load_in_8bit=True`
   - 如果还不够，继续使用云端 API

### 如果你想微调：

1. **准备数据**：
   - 运行 `prepare_vlx_finetuning_data.py`
   - 收集至少 500 个标注样本

2. **选择平台**：
   - AutoDL：租一块 RTX 3090（2元/时）
   - 预计 2-4 小时完成 QLoRA 微调

3. **验证效果**：
   - 在测试集上评估
   - 对比 API 基线
   - 导出权重供本地使用

---

## 九、总结

| 问题 | 答案 |
|------|------|
| **能本地部署 3B 吗？** | 勉强可以（需要量化） |
| **8 GB 显存够吗？** | 推理：临界；微调：不够 |
| **LLamaFactory 支持吗？** | 支持（基于 Qwen2-VL） |
| **能本地微调吗？** | 不建议（需要 QLoRA + 云端 GPU） |
| **最佳方案？** | **继续用 API + 必要时云端微调** |

---

**建议**：当前阶段继续使用 Om AI Lab 的云端 API，专注于实验设计和数据收集。等积累足够数据（500+ 样本）且确认需要定制化后，再考虑云端微调。

---

**参考资料**：
- [VLX-Seek GitHub](https://github.com/om-ai-lab/VLX-Seek)
- [LLamaFactory 文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2-VL 微调指南](https://qwen.readthedocs.io/)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)

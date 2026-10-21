# VLX-Seek API 集成指南

本文档说明如何使用 Om AI Lab 的云端 VLX-Seek API 替代本地推理。

## 背景

原 `vlx_seek_probe.py` 设计为调用本地 VLX-Seek 模型（约 20 GB 权重文件），但本机只有 8 GB GPU 显存。通过接入 Om AI Lab 的云端 API，可以在不下载模型的情况下完成推理。

## API 服务信息

- **提供商**：Om AI Lab
- **模型选项**：
  - `vlx-seek`（原始模型）
  - `vlx-seek-1.5-3b`（推荐，新版本，更适合具身场景）
- **认证**：Bearer Token
- **请求格式**：JSON + 图像 URL（当前实现使用 base64 data URL）
- **返回格式**：JSON，包含检测框和标签

## 环境配置

### 1. 设置环境变量

在 PowerShell 中设置 API 凭证：

```powershell
# 设置 API Key（从 Om AI Lab 平台获取）
$env:OM_API_KEY = "your-api-key-here"

# 设置 API Endpoint（从平台的"鉴权信息"页面复制）
$env:OM_API_ENDPOINT = "https://your-endpoint-url"
```

**持久化设置**（可选，使环境变量在新会话中保留）：

```powershell
[System.Environment]::SetEnvironmentVariable("OM_API_KEY", "your-api-key-here", "User")
[System.Environment]::SetEnvironmentVariable("OM_API_ENDPOINT", "https://your-endpoint-url", "User")
```

### 2. 安装依赖

API 客户端需要 `requests` 库：

```powershell
& .\.venv\python.exe -m pip install requests
```

## 快速测试

运行测试脚本验证 API 连接：

```powershell
& .\.venv\python.exe .\python\test_vlx_api_client.py
```

**成功输出示例**：

```
VLX-Seek API Client Test Suite
============================================================
Testing API connection...
✓ API key found: sk_xxxxx...
✓ API endpoint: https://api.example.com
✓ Client initialized with model: vlx-seek-1.5-3b

============================================================
Testing VLX-Seek API with synthetic image...
============================================================

Image size: (600, 400)
Number of proposals: 3
Task description: 'the rightmost red rectangle'

Calling VLX-Seek API...

------------------------------------------------------------
API Response:
------------------------------------------------------------
Answer: <ground>the rightmost red rectangle</ground><objects><obj2></objects>

Detected boxes: 1

[0] <obj2>
    Box: [450, 100, 550, 200]
    Label: the rightmost red rectangle

------------------------------------------------------------
Usage Statistics:
------------------------------------------------------------
Total requests: 1
Prompt tokens: 1142
Completion tokens: 18
Total tokens: 1160
Total time: 2159 ms
Average time per request: 2159.0 ms

============================================================
✓ All tests passed!
```

## 运行海事图像回放（使用 API）

### 方式 1：直接使用 API 客户端

使用独立脚本（需要创建）：

```powershell
& .\.venv\python.exe .\python\maritime_vlx_api_replay.py `
  --report .\logs\maritime_single_20260925_205856_936729\report.json `
  --output .\logs\vlx_api_result.json `
  --proposal-mode yolo_red_regions `
  --model vlx-seek-1.5-3b
```

### 方式 2：修改现有 vlx_seek_probe.py

在 `vlx_seek_probe.py` 中添加 `--use-api` 参数：

```powershell
& .\.venv\python.exe .\python\vlx_seek_probe.py `
  --report .\logs\maritime_single_20260925_205856_936729\report.json `
  --output .\logs\vlx_api_result.json `
  --proposal-mode yolo_red_regions `
  --run-vlx --use-api `
  --model vlx-seek-1.5-3b
```

**参数说明**：

- `--use-api`：使用云端 API 而非本地模型
- `--model`：选择模型 ID（`vlx-seek` 或 `vlx-seek-1.5-3b`）
- 移除 `--vlx-repo` 和 `--model-path`（API 模式不需要）

## API 客户端使用示例

### Python 代码示例

```python
from PIL import Image
from vlx_api_client import VLXSeekAPIClient

# 初始化客户端
client = VLXSeekAPIClient(
    api_key="your-api-key",  # 或从环境变量读取
    api_endpoint="https://your-endpoint",
    model="vlx-seek-1.5-3b"
)

# 加载图像
image = Image.open("path/to/image.png").convert("RGB")

# 定义候选框（来自 YOLO 或其他检测器）
proposals = [
    [50, 100, 150, 200],   # [x1, y1, x2, y2]
    [250, 100, 350, 200],
    [450, 100, 550, 200]
]

# 调用 API
result = client.ground(
    image=image,
    boxes=proposals,
    description="the rightmost red balloon in the image",
    lang="en",
    temperature=0.0,
    max_new_tokens=128
)

# 处理结果
print(f"Answer: {result['answer']}")
for bbox in result['result_bbox_list']:
    print(f"Detected: {bbox['object_index']} at {bbox['xmin']}, {bbox['ymin']}, {bbox['xmax']}, {bbox['ymax']}")

# 查看用量统计
usage = client.get_usage_summary()
print(f"Total tokens: {usage['total_tokens']}")
print(f"Average latency: {usage['average_took_ms']:.1f} ms")
```

## 两阶段运行（缓存候选框）

与本地模式相同，可以先生成候选框，再调用 API：

**第一步：生成 YOLO 候选框**（本地，不调用 API）

```powershell
& .\.venv\python.exe .\python\vlx_seek_probe.py `
  --report .\logs\maritime_single_20260925_205856_936789\report.json `
  --output .\logs\proposals_cache.json `
  --proposal-mode yolo_red_regions
```

**第二步：使用缓存的候选框调用 API**

```powershell
& .\.venv\python.exe .\python\vlx_seek_probe.py `
  --report .\logs\maritime_single_20260925_205856_936789\report.json `
  --output .\logs\vlx_api_result.json `
  --proposals-report .\logs\proposals_cache.json `
  --proposal-mode yolo_red_regions `
  --run-vlx --use-api `
  --model vlx-seek-1.5-3b
```

**优势**：

- 第一步可以在本地 8GB GPU 上完成
- 第二步不需要 YOLO 权重文件
- 可以在不同机器上分别执行

## 模型选择建议

### vlx-seek-1.5-3B（推荐）

- **优势**：
  - 针对具身场景优化
  - 更适合机器人/无人机任务
  - 提示词模板简洁
- **提示词示例**：
  ```
  识别每个：最右边的红色气球。
  识别每个：没有盖瓶盖的水瓶。
  ```

### vlx-seek（原始模型）

- **优势**：
  - 支持更多任务类型（区域文字识别、区域问答等）
  - 更详细的推理过程
- **提示词示例**：
  ```
  请在图像中检测 最右边的红色气球，并用对象索引回答这个问题。
  ```

## 成本估算

根据 API 文档：

```
消耗额度 = (输入 Token + 输出 Token) × 模型系数 / 1000
```

**单帧估算**（基于测试数据）：

- 输入：约 1140 tokens（图像 + 提示词）
- 输出：约 20 tokens（检测结果）
- 总计：约 1160 tokens/帧

**30 帧回放**：

- 总 tokens：约 34,800
- 具体费用取决于平台的模型系数和定价

**建议**：

1. 先用 5 帧测试，验证效果
2. 在平台的"模型用量和性能监控"查看实际消耗
3. 确认效果后再批量处理全部帧

## 与本地推理的对比

| 维度 | 本地推理 | 云端 API |
|------|---------|---------|
| **显存需求** | ~20 GB | 0（云端运行） |
| **网络** | 不需要 | 需要稳定网络 |
| **延迟** | 低（本地） | 中（网络 + 云端） |
| **成本** | 硬件投入 | 按 token 计费 |
| **环境配置** | 复杂（Transformers、PyTorch） | 简单（仅需 requests） |
| **适用场景** | 大量推理、无网络 | 快速验证、硬件受限 |

## 常见问题

### Q1: API 返回 401 Unauthorized

**原因**：API Key 无效或未设置

**解决**：

```powershell
# 检查环境变量
echo $env:OM_API_KEY

# 重新设置
$env:OM_API_KEY = "your-correct-api-key"
```

### Q2: API 返回 403 Forbidden 或"模型无权限"

**原因**：当前账户未开通该模型

**解决**：

1. 登录 Om AI Lab 开发者平台
2. 进入"模型广场"
3. 确认 `VLX-Seek` 或 `VLX-Seek-1.5-3B` 已开通

### Q3: 图像过大导致请求超时

**原因**：当前实现使用 base64 编码，图像过大时请求体膨胀

**解决**：

1. 调整超时参数：
   ```python
   client = VLXSeekAPIClient(timeout=120)  # 默认 60 秒
   ```

2. 图像预处理（降低分辨率）：
   ```python
   max_size = 1280
   if max(image.size) > max_size:
       image.thumbnail((max_size, max_size), Image.LANCZOS)
   ```

### Q4: 返回的框不在输入候选框中

**原因**：模型可能返回新检测的框，而非从候选中选择

**解决**：

- 当前 `vlx_api_client.py` 会抛出 `ValueError`
- 如果是 VLX-Seek 的预期行为，可以修改 `validated_vlx_box()` 的容错逻辑

### Q5: 如何切换回本地推理？

**方法**：移除 `--use-api` 参数，恢复 `--vlx-repo` 和 `--model-path`：

```powershell
& .\.venv\python.exe .\python\vlx_seek_probe.py `
  --report <report> `
  --output <output> `
  --proposal-mode yolo_red_regions `
  --run-vlx `
  --vlx-repo .\downloads\VLX-Seek `
  --model-path .\models\VLX-Seek-1.5-10B
```

## 下一步

1. **运行基础测试**：
   ```powershell
   & .\.venv\python.exe .\python\test_vlx_api_client.py
   ```

2. **小批量验证**（5 帧）：
   创建只包含 baseline 条件的临时报告，测试 API 效果

3. **对比 YOLO 基线**：
   - YOLO 最高置信度选择
   - YOLO 最右气球规则
   - VLX-Seek API 推理

4. **全量运行**（30 帧）：
   确认效果和成本后，处理所有六种条件

5. **人工复核**：
   特别是 `target_occluded` 条件下的遮挡检测

## 技术细节

### 接口映射

| VLXSeekWorker（本地） | VLXSeekAPIClient（云端） |
|---------------------|------------------------|
| `__init__(model_path, device)` | `__init__(api_key, api_endpoint, model)` |
| `ground(image, boxes, description, ...)` | `ground(image, boxes, description, ...)` |
| 返回格式 | **相同**（兼容现有评估代码） |

### 图像传输方式

当前实现：**base64 data URL**

- 优势：无需额外的文件上传接口
- 劣势：请求体较大（~33% 膨胀）

**未来改进**：如果 Om AI Lab 提供专用的图像上传接口，可以：

1. 先上传图像获取 URL
2. API 请求中只传 URL
3. 减少请求体大小，提升速度

### 提示词模板

API 客户端内置了两个模型的提示词模板：

- **vlx-seek-1.5-3b**：`"识别每个：{description}。"`
- **vlx-seek**：`"请在图像中检测 {description}，并用对象索引回答这个问题。"`

模板来自官方文档，保证与云端模型训练时的格式一致。

## 参考资料

- [Om AI Lab 开发者平台](https://platform.om-agent.cn/)
- [VLX-Seek API 文档](https://s.apifox.cn/46c7cae8-f2d7-4780-a960-e531d6c5eaa8)
- [VLX-Seek 研究博客](https://om-ai-lab.github.io/)
- 本项目原文档：`docs/VLX-Seek接入预检与海事回放.md`

---

**文档版本**：v1.0（2026-09-27）  
**作者**：AI 辅助生成  
**状态**：API 客户端已实现，集成到 `vlx_seek_probe.py` 待完成

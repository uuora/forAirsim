# VLX-Seek 交互式演示指南

## 🎮 概述

这是一个仿照 VLX-Seek 论文中的交互式演示界面，可以：

1. 加载任意图像
2. 手动框选候选区域
3. 输入自然语言描述
4. VLX-Seek API 自动识别目标

---

## 🚀 快速启动

### 方法 1：双击启动脚本

```
Run-VLX-Demo.cmd
```

会自动加载一张海事场景图像。

### 方法 2：命令行启动

```powershell
# 不加载图像（空白界面）
& .\.venv\python.exe .\python\vlx_demo_ui.py

# 加载指定图像
& .\.venv\python.exe .\python\vlx_demo_ui.py --image "path/to/image.png"

# 使用原版 VLX-Seek 模型（默认是 1.5-3B）
& .\.venv\python.exe .\python\vlx_demo_ui.py --model vlx-seek
```

---

## 🎯 使用步骤

### 1. 加载图像

**首次启动**：
- 如果使用启动脚本，会自动加载示例图像
- 如果空白启动，按 `N` 键加载图像

**更换图像**：
- 按 `N` 键
- 在控制台输入图像路径（支持拖拽）

### 2. 绘制候选框

**操作**：
- 左键点击并拖拽，绘制矩形框
- 可以绘制多个候选框
- 绿色边框表示候选区域，带编号

**提示**：
- 候选框应该覆盖所有可能的目标对象
- 也可以包含干扰物（VLX 会筛选）
- 至少绘制 1 个候选框

### 3. 输入描述并运行

**操作**：
- 按 `R` 键
- 在控制台输入自然语言描述
- 等待 VLX-Seek API 返回结果

**描述示例**：

**英文**：
```
the rightmost red balloon
the balloon without a cap
the person wearing a blue shirt
the car on the left side
the second bottle from right to left
```

**中文**（支持）：
```
最右边的红色气球
没有瓶盖的水瓶
穿蓝色衣服的人
左侧的汽车
从右往左数第二瓶水
```

### 4. 查看结果

- VLX 选中的目标会用**红色粗边框**高亮
- 顶部显示识别结果
- 控制台输出详细信息

---

## ⌨️ 快捷键

| 按键 | 功能 |
|------|------|
| **鼠标拖拽** | 绘制候选框 |
| **C** | 清除所有候选框 |
| **R** | 运行 VLX-Seek（输入描述） |
| **N** | 加载新图像 |
| **Q** 或 **ESC** | 退出程序 |

---

## 📸 示例场景

### 场景 1：基线条件

**图像**：`logs/maritime_single_20260925_205856_936729/baseline_000.png`

**候选框**：框选所有可见的红色气球

**描述**：
```
the rightmost red balloon in the image
```

**预期结果**：✓ 选中右侧的目标气球

---

### 场景 2：相似干扰

**图像**：`logs/maritime_single_20260925_205856_936729/similar_red_000.png`

**候选框**：框选两个红色气球

**描述**：
```
the rightmost red balloon
```

**预期结果**：✓ 准确区分左右两个气球

---

### 场景 3：遮挡条件

**图像**：`logs/maritime_single_20260925_205856_936729/target_occluded_000.png`

**候选框**：框选可见的红色区域（可能不完整）

**描述**：
```
the rightmost red balloon
```

**预期结果**：✓ 即使部分遮挡，仍能识别

---

### 场景 4：属性推理

**自定义图像**：包含多个不同状态的对象

**候选框**：框选所有水瓶

**描述**：
```
the water bottle without a cap
```

**预期结果**：VLX 根据属性识别目标

---

### 场景 5：空间关系

**自定义图像**：一排物体

**候选框**：框选所有物体

**描述**：
```
the second bottle from left to right
```

**预期结果**：VLX 理解顺序关系

---

## 🎨 界面说明

### 显示元素

1. **绿色边框** - 你绘制的候选框
2. **黄色边框** - 正在绘制的框（临时）
3. **红色粗边框** - VLX-Seek 选中的目标
4. **顶部文字** - 识别结果文字
5. **底部说明** - 快捷键提示

### 控制台输出

运行 VLX 时，控制台会显示：
```
Running VLX-Seek...
  Description: the rightmost red balloon
  Boxes: 3
  Answer: <ground>the rightmost red balloon</ground><objects><obj2></objects>
  Detected: 1 objects
```

---

## 🔧 高级用法

### 批量测试不同描述

创建测试脚本：

```python
from vlx_demo_ui import VLXDemoUI
from PIL import Image

demo = VLXDemoUI()
demo.load_image("path/to/image.png")

# 手动添加候选框
demo.boxes = [
    [50, 100, 150, 200],
    [250, 100, 350, 200],
    [450, 100, 550, 200]
]

# 测试多个描述
descriptions = [
    "the leftmost object",
    "the middle object",
    "the rightmost object"
]

for desc in descriptions:
    print(f"\nTesting: {desc}")
    demo.run_vlx(desc)
```

### 使用自己的检测器

替换手动绘制，用 YOLO 自动生成候选框：

```python
from yolo_detector import YoloTargetDetector

# 加载 YOLO
detector = YoloTargetDetector("models/yolo11n.pt")

# 生成候选框
image = cv2.imread("test.png")
result = detector.detect(image)

# 转换为候选框列表
demo.boxes = [c["bbox_xyxy_pixels"] for c in result["candidates"]]

# 运行 VLX
demo.run_vlx("your description here")
```

---

## 💡 提示与技巧

### 1. 如何写好描述？

**✅ 好的描述**：
- 具体明确：`the rightmost red balloon`
- 包含属性：`the water bottle without a cap`
- 空间关系：`the person on the left side`
- 顺序关系：`the second car from right`

**❌ 不好的描述**：
- 太模糊：`the thing`
- 无区分度：`the balloon`（有多个气球时）
- 不在图中：`the cat`（图中没有猫）

### 2. 候选框绘制技巧

- **覆盖目标**：框要完全包含目标，但不要太大
- **包含干扰**：可以框选相似的干扰物，VLX 会筛选
- **遮挡情况**：框选可见部分即可

### 3. 常见问题

**Q: 描述输入中文有效吗？**

A: 有效！VLX-Seek 支持中文描述。但推荐用英文，因为模型主要在英文数据上训练。

**Q: 候选框必须精确吗？**

A: 不需要。VLX 会对候选框做微调（±几个像素）。

**Q: 可以不画候选框，直接让 VLX 检测吗？**

A: 不行。VLX-Seek 的设计是 **grounding**（从候选框中选择），而非 **detection**（从零检测）。这是它与 YOLO 等检测器的区别。

**Q: API 调用失败怎么办？**

A: 检查：
1. `.env` 文件中的 API 凭证是否正确
2. 网络连接是否正常
3. API 额度是否充足

---

## 📊 性能对比

### 海事场景测试结果（30 帧）

| 方法 | 总体准确率 | 遮挡条件 | 相似干扰 |
|------|-----------|---------|---------|
| YOLO 最高置信 | 53% | 0/5 | 1/5 |
| 简单最右规则 | 83% | 0/5 | 5/5 |
| **VLX-Seek API** | **100%** | **5/5** | **5/5** |

VLX-Seek 在**遮挡**和**相似干扰**场景下显著优于基线方法。

---

## 🛠️ 故障排除

### 图像无法加载

**症状**：按 N 键后提示"Cannot load image"

**解决**：
- 检查路径是否正确（支持拖拽）
- 确保是支持的格式（PNG, JPG, JPEG）
- 使用绝对路径或相对于项目根目录的路径

### API 调用超时

**症状**：长时间等待无响应

**解决**：
- 检查网络连接
- 图像过大时，程序会自动缩放到 1280px
- 增加超时设置（修改 `vlx_api_client.py` 中的 `timeout`）

### 候选框无法绘制

**症状**：鼠标拖拽无效

**解决**：
- 确保窗口是激活状态
- 框的宽高至少 10 像素
- 重新启动程序

---

## 📚 相关资源

- **API 文档**：`docs/VLX-API集成指南.md`
- **部署指南**：`docs/VLX-Seek-1.5-3B本地部署与微调.md`
- **测试结果**：`logs/vlx_api_full_test.json`
- **可视化对比**：`logs/vlx_api_visualizations/`

---

## 🎓 论文参考

**VLX-Seek 系列论文**：
- [VLX-Seek: VLM 细粒度感知增强](https://om-ai-lab.github.io/2026_06_27_vlx_seek_zh.html)
- [VLX-Seek 1.5: 面向具身场景的细粒度感知增强](https://om-ai-lab.github.io/2026_07_06_vlx_seek_1_5_zh.html)

**核心能力**：
- 区域级视觉推理（Region-based Visual Reasoning）
- 细粒度目标定位（Fine-grained Object Grounding）
- 语言引导的视觉选择（Language-guided Visual Selection）

---

## 🆕 下一步

体验完演示后，你可以：

1. **查看可视化对比**：
   ```
   explorer logs\vlx_api_visualizations
   ```

2. **尝试自己的图像**：
   - 拍摄或下载测试图像
   - 按 N 键加载
   - 设计不同的描述进行测试

3. **扩展功能**：
   - 集成到你的应用中
   - 批量处理数据集
   - 评估不同模型版本

4. **准备微调数据**：
   - 记录 VLX 失败的案例
   - 收集你的特定领域数据
   - 参考 `docs/VLX-Seek-1.5-3B本地部署与微调.md`

---

**版本**：v1.0  
**日期**：2026-09-28  
**作者**：AI 辅助生成

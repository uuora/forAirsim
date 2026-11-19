# TOPPQuad · AirSim 轨迹控制

**RotorPy 实验**　[运行说明](docs/使用说明/TOPPQuad三基线仿真.md#当前-rotorpy-随机航路批次) · [实验结果](reports/rotorpy_current/conclusion.md) · [论文式图集 PDF](reports/rotorpy_current/paper_figures/paper_figures.pdf) · [本机三维回放](http://127.0.0.1:8765/)

双击 [打开几何控制实验.cmd](打开几何控制实验.cmd)。控制台默认加载 Bézier、Hermite、B 样条三条 TOPPQuad 参考，选择其中一种后可手动开始验证。现有 AirSim 实例可通过“显示 UE 视口”观察。

- **弹出三维地图**会把当前地图移到独立、可缩放的窗口。关闭地图窗口或点击控制台的“收回三维地图”即可还原。
- 地图默认按航线取景，显示跟踪阶段的轨迹。勾选“含起降”可看全程实测和完整高度范围。高度 ×1.5 和 ×2 仅影响三维显示。XY 俯视和 X / 高度剖面使用等比例。
- 原三种基线及 E0–E4 任务可用 `./打开几何控制实验.cmd -Baseline` 打开。添加 `-DetachedMap` 可直接分窗。

| 目录 | 内容 |
| --- | --- |
| `python/geometric_validation` | 路径和时间参数化、Qt 地图、AirSim 跟踪与独立审计 |
| `configs` | 三种基线、TOPPQuad 冻结参考、AirSim 设置 |
| `.runtime/toppquad_upstream` | 固定版本的 TOPPQuad 上游 Git 源码 |
| `reports`、`logs` | 当前 TOPPQuad 批量结果和单次飞行原始记录 |

在本机，`external/AirSim` 与 `.venv` 复用 `D:\forAirsim` 的仿真和 Python 环境。其他机器需准备相同的 AirSim/UE 构建与 Python 依赖，详见 [AirSim 补丁](patches/README.md) 和 [运行与复现](docs/使用说明/TOPPQuad三基线仿真.md)。

[TOPPQuad 实测对比](reports/geometric_three_planners_toppquad_20261119_v1/index.html) · [数据字典](docs/使用说明/几何控制数据字典.md)

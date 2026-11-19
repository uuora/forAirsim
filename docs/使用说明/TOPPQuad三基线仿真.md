# TOPPQuad 与三种轨迹基线

开发分支为 `codex/toppquad-baselines`。首轮覆盖默认三维连续过点任务，分别对冻结的 Bézier、Hermite、B 样条路径进行时间重参数化。Bézier 与 Hermite 基线的几何几乎等价。

## 启动 Qt

在该分支工作树中运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/open-geometric-control-panel.ps1
```

默认加载 TOPPQuad 冻结配置。选择一种方法后，按控制台提示连接或启动 AirSim，手动开始仿真。运行 Qt 和飞行无需安装 CasADi。追加 `-Check` 可离屏核验面板，`-Baseline` 打开原三种基线。

点“弹出三维地图”可把同一地图移到独立窗口，继续同步路线、飞机姿态与实测轨迹。地图窗口可自由缩放，关闭它会把地图收回控制台。可用 `-DetachedMap` 直接分窗启动。

三维地图默认按航线取景、等比例投影，实测线显示跟踪阶段。勾选“含起降”可看全程实测和完整高度范围。选择“高度 ×1.5”或“高度 ×2”可放大三维投影的竖向变化。图上高度读数始终是原始米制坐标，XY 和 X / 高度保持等比例。三维左拖旋转，右拖平移，滚轮缩放。

## 重新求解及运行批次

在已经安装几何控制依赖的 Python 环境中执行：

```powershell
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/maokat12/lbTOPPQuad.git .runtime/toppquad_upstream
git -C .runtime/toppquad_upstream sparse-checkout set --cone TOPPQuad
git -C .runtime/toppquad_upstream checkout 3db181cb0c76baaa1669f859f16a83060c2ee7a0
python -m pip install casadi==3.7.2
python python/run_geometric_planner_comparison.py --toppquad-upstream .runtime/toppquad_upstream --config configs/geometric_planners_v1.json --output configs/geometric_toppquad_rebuilt.json
python python/run_geometric_planner_comparison.py --config configs/geometric_toppquad_rebuilt.json --airsim --launch
```

`D:/TOPPQuad` 本机已有上游源码。使用随目录保存的 CasADi 时，在重新求解前可设置 `$env:PYTHONPATH = (Resolve-Path .runtime/toppquad_deps).Path`，并使用 `./.venv/python.exe`。

生成配置时要求输出文件不存在。批次每种方法飞行一次，保留状态、命令、过点核验、独立审计、降落回执与图表。上一项的降落、解除武装或 API 释放未确认时，后续项目保留为未运行。

## 求解与执行边界

使用 Mao 等人公开的优化版 TOPPQuad，固定为上述 Git 提交。冻结路径在 61 个等间隔的原参考时间节点上采样，以 ENU 送入求解器。求解阶段使用上游 Crazyflie 参数。导出阶段用单调五次多项式平滑离散时间映射，对原曲线做解析复合，并统一拉长时间以满足现有速度、加速度和 jerk 门槛。冻结配置保存原始节点时间、进度速率平方、各旋翼推力、拟合误差、时间拉伸因子和上游源码哈希。 [作者仓库](https://github.com/maokat12/lbTOPPQuad)

AirSim 使用 SimpleFlight 与原 Qt 几何跟踪器，跟踪时沿用基线固定偏航。上游预测旋翼推力仅作求解记录。实验对实际过点和跟踪可行性进行核验，原始求解时长、导出参考时长与实测完成时长分别记录。实验为纯控制连续过点，关闭相机截图。

## 2026-11-19 首轮 AirSim 结果

| 路径 | 原基线参考（秒） | TOPPQuad 离散解（秒） | 导出参考（秒） | 实测完成（秒，含终点停稳） | 峰值位置误差（米） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bézier | 10.737 | 7.814 | 13.229 | 14.598 | 0.110 |
| Hermite | 10.737 | 7.814 | 13.229 | 14.616 | 0.111 |
| B 样条 | 11.493 | 7.744 | 13.679 | 14.988 | 0.082 |

三条同条件记录均通过正常完成、跟踪、过点、独立数据审计与安全降落验收。受平滑与原参考门槛影响，三条最终导出参考均比原基线慢。每种方案仅一次，当前结果支持该任务下的接入可行性。原始状态、命令、回执和派生图表见新工作区的 [实测对比页](../../reports/geometric_three_planners_toppquad_20261119_v1/index.html)。

## 当前 RotorPy 随机航路批次

[打开实验入口](../../RotorPy仿真实验.html) · [实时进度与三维回放](http://127.0.0.1:8765/)。

当前入口为 python/run_rotorpy_campaign.py。首次运行从系统随机源生成 100 条航路，并先保存 reports/rotorpy_current/design.json。之后继续运行使用该文件中的同一批航点。三种路径构造分别比较稳健计时＋航向跟踪、速度 TOPP、速度＋总推力 TOPP 和 TOPPQuad，共 1,200 项。各组使用同一航向函数，统一比较首飞。规划失败、仿真失败和通过均有独立回执。

当前看板以参考逐电机推力、参考飞行时间和闭环完成时间为主。闭环完成时间及跟踪误差按同航路双方正常完成的样本配对。位置、过点速度、倾角、机头误差和整体姿态等门槛保留为逐项比较数据。刷新统计使用已有数据。

[论文式图集](http://127.0.0.1:8765/paper_figures/)提供按本项目三种路径组织的推力箱线图、参考时间箱线图和同路线时间缩短率直方图，并附离散解与连续参考的计时差。用 python/run_rotorpy_paper_figures.py 可从原记录重新导出 PNG、SVG 和整套 PDF。

### 查看当前批次

双击根目录的 [RotorPy仿真实验.html](../../RotorPy仿真实验.html)，或打开 [当前看板](http://127.0.0.1:8765/)。若本地服务尚未启动，在 VS Code 中打开本项目文件夹，按 Ctrl+Shift+P 搜索“运行任务”，选择“RotorPy · 打开本地实验服务”。“RotorPy · 100 条随机航路 · 运行或继续”使用默认目录 reports/rotorpy_current，已有设计时按其中的航点继续。

### 新跑一批并打开结果

在项目根目录的 PowerShell 终端执行。每批使用新的目录名。中断后用同一个目录名再执行，脚本按已保存的航点继续缺少回执的工况。

~~~powershell
$batch = 'reports/rotorpy_batch_01'
$env:PYTHONPATH = '.runtime/toppquad_deps;.runtime/rotorpy_deps;python'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\python.exe -X utf8 -u python/run_rotorpy_campaign.py --output $batch --jobs 8
~~~

可在第一轮命令末尾加 --pilot，先验证 3 条航路、36 项工况。随后用同一目录运行完整命令便会补齐全批。出现 finished.json 后，可在同一个终端生成图集。

~~~powershell
.\.venv\python.exe -X utf8 python/run_rotorpy_paper_figures.py --source $batch
~~~

另开终端，在项目根目录为该批启动服务。保持该终端运行，然后打开 [新批次看板](http://127.0.0.1:8766/) 或 [新批次图集](http://127.0.0.1:8766/paper_figures/)。

~~~powershell
.\.venv\python.exe -u python/serve_rotorpy_paper.py reports/rotorpy_batch_01 --port 8766
~~~

批次目录中的 design.json 保存 100 条航路和协议，runs.csv 为逐次汇总，paired.csv 为同航路配对，waypoints.csv 为逐航点数据。三种 CSV 可用 Excel 或 VS Code 打开。逐条时序在 route_编号/路径构造/方法/ 下，simulation.csv.gz 是 50 Hz 实测时序，reference_metrics.csv 是参考时序，heading.csv 是机头时序。三维看板可按路线、构造和方法直接选择对应的数据。

# TOPPQuad 与三种轨迹基线

开发分支为 `codex/toppquad-baselines`。首轮覆盖默认三维连续过点任务，分别对冻结的 Bézier、Hermite、B 样条路径进行时间重参数化。Bézier 与 Hermite 基线的几何几乎等价。

## 启动 Qt

在该分支工作树中运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/open-geometric-control-panel.ps1 -ToppQuad
```

选择带 TOPPQuad 标签的方法后，按原界面操作连接或启动 AirSim，并开始仿真。这个入口使用随分支保存的冻结配置，运行面板及仿真无需安装 CasADi。追加 `-Check` 可离屏核验面板和方法切换。

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

三条同条件记录均通过正常完成、跟踪、过点、独立数据审计与安全降落验收。受平滑与原参考门槛影响，三条最终导出参考均比原基线慢。每种方案仅一次，当前结果支持该任务下的接入可行性。原始状态、命令、回执和派生图表在主工作区的 `reports/geometric_toppquad_20261119_v1/`。

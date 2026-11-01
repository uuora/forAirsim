# AirSim方法选择与结果对比

2026-10-30更新：按用户澄清，当前默认已经切换为03地图中的无障碍物纯控制实验，并加入固定/随机目标与选定方法连续测试。当前操作请优先阅读[纯控制六组仿真使用说明](纯控制六组仿真_使用说明_20261030.md)；下文保留早期冻结轨迹回放接口的说明与历史核验记录。

本面板用于选择已有轨迹方案中的一个方法进行AirSim执行，然后比较明确选中的历史结果。启动面板本身不启动飞行。当前仍为“简化模型搜索后，AirSim执行最终参考”的实验方式。

## 打开面板

在 `D:\forAirsim` 双击 `打开轨迹实验面板.cmd`；或在PowerShell运行：

```powershell
Set-Location D:\forAirsim
& .\scripts\open-curve-airsim-panel.ps1
```

1. 在“方法”下拉框中选择普通Bézier、普通Hermite，或其ABOM/改进候选版本。
2. 点击“运行所选方法”。每次只运行当前方法；需要运行另一组时再手动选择并启动。
3. 勾选“显示AirSim窗口”可观察本次执行；未勾选时后台运行。每次使用独立结果目录，不覆盖旧记录。
4. 完成后，勾选“历史结果”左侧要比较的记录，点击“比较选中结果”；也可按Ctrl/Shift多选。勾选项优先，可用“清除勾选”重置。
5. 对比目录包含CSV、JSON、中文说明以及轨迹、误差和指标图，可用“打开最近对比”查看。

普通曲线指当前高阶端状态保持参考，并非早期三次Bézier版本。改进组仍标为候选方案，其效果需由多场景实验检验。

## 对比的规则

- 只比较用户明确选择的“运行批次＋方法”组合，其他方法不自动加入。
- 同一方法多次运行逐条保留，并报告重复次数；单次结果不产生显著性或置信区间结论。
- 物理场景、控制器、初始化、误差阈值、曲线实现和原轨迹方案等条件不同的记录会拒绝合并。显示窗口等展示选项单独记录。
- “流程完成”与“控制误差通过”分别显示；失败和未运行槽位不伪造指标。
- 对比反映执行误差等实测指标，不据此推断严格规划墙钟速度优势。

已提供[三种Hermite方法的历史对比示例](../../reports/airsim_compare_selection_20261029_v1/comparison.md)。它使用现有真实AirSim记录，三组当时均未通过控制误差门槛。

## 命令行按方法选择

启动器不带组别时会显示编号菜单，等待选择后才运行。指定单组可直接执行：

```powershell
& .\scripts\run-airsim-curve-replay-pilot.ps1 -Groups hermite_abom -ShowWindow
```

明确选择两个组：

```powershell
& .\scripts\run-airsim-curve-replay-pilot.ps1 -Groups @('hermite_ordinary','hermite_abom') -ShowWindow
```

只有显式传入 `-AllGroups` 才会自动执行全部六组。`-ProbeOnly` 仅做地面接口探针。

启动器输出直接写入 `reports/airsim_selected_时间戳.launcher.log`，面板只读取并显示日志。关闭面板不向独立实验发送停止指令，也不关闭它的日志写入端；重新打开面板可刷新已保存的历史记录。专用端口41461被占用时，新的运行会拒绝启动，保留已有实例。

## 输出和代码

- [选择面板](../../python/curve_airsim_experiment_panel.py)
- [独立实例启动器](../../scripts/run-airsim-curve-replay-pilot.ps1)
- [按组选择的执行器](../../python/run_airsim_curve_replay_pilot.py)
- [独立核查器](../../python/audit_airsim_curve_replay_pilot.py)
- [历史结果比较器](../../python/compare_airsim_curve_runs.py)

新增运行保存在 `reports/airsim_selected_时间戳/`；新增对比保存在 `reports/airsim_comparison_时间戳/`。轨迹方案可在面板中切换文件，默认使用当前已生成的六项方案。AI辅助实现与核查。

## 本次功能验证

- 已完成真实AirSim单组运行：`airsim_selected_smoke_20261029_v1`，仅执行Hermite改进候选，其余五组未加入运行。独立核查为`PASS / SELECTED_SUBSET`；该次控制误差门槛未通过，保留原始结果。
- 已通过面板勾选上述单组记录与历史Hermite基础ABOM记录生成对比：`airsim_comparison_20261029_231829_751896`，核实仅含两条指定记录，输出三类图的PNG/PDF及逐条指标。
- 已核验下拉选择不会自动启动仿真，勾选列可直接用鼠标操作；界面保持可用。

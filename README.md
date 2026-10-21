# AirSim dual-drone balloon simulation

**从 GitHub 恢复本地功能**：初始化 AirSim 子模块后，按 [AirSim 本地补丁说明](patches/README.md) 应用 C++ 与自建场景补丁，再构建项目。API 凭证使用本地 `.env` 配置；实验输出、模型和构建缓存保留在本地。

**当前控制任务（2026-10-14）**：[03_三维双目标控制.cmd](03_三维双目标控制.cmd)启动独立的单机三维控制窗口。默认场地为 100×100 米、高度 2～25 米；直接读取红蓝目标坐标，联合规划两点并滚动补充目标。“查看 UE4 场景”按需显示同一 AirSim 的原生三维场景，默认近距离跟随，支持鼠标旋转、平移、缩放及聚焦与整体场景按钮；“隐藏 UE4 场景”恢复后台主视口不渲染，飞行继续；俯视图作为可选补充。使用轻量控制地图和端口 41460，打开窗口不会自动起飞。操作与配置见[三维双目标控制](docs/三维双目标控制.md)，实机背景见[推重比与穿越机迁移](docs/推重比与穿越机迁移.md)。下方为此前视觉任务记录。

当前实验入口（2026-10-12）：[01_固定地图实验.cmd](01_固定地图实验.cmd)用于 v24 视觉控制闭环任务窗口，UE 在后台正常渲染，打开窗口不会自动起飞；[02_识别与回放工作台.cmd](02_识别与回放工作台.cmd)用于识别与历史回放。根目录保留6个中文入口，其他4个为地图维护工具，详见[命令使用说明](命令使用说明.md)。下文旧阶段记录保留；其中已退役的英文入口不再位于根目录。


**继续推进（2026-10-07，采样成本定位）**：已复算813主帧、7缺槽，最新重复的主线程处理95分位数453毫秒；下一轮补细分计时、有界落盘和停稳证明。Qt增加重复编号及日期，14项Qt复验通过。[采样成本与下一轮修复计划](D:/forAirsim/docs/research-plan-20261004/任务一采样成本定位与下一轮修复计划.md)。

**最新推进（2026-10-07，尺度验证与Qt5工作台）**：50毫米独立重复未达标，100毫米条件未执行。保留全部5个尝试，完整会话保存813主帧，失败/中断单列；431项回归通过。Qt5现支持中文实验回放、图表与导出，并保留真实YOLO/VLX单图验证。完整几何和正式对照仍待完成。[结果与下一步](D:/forAirsim/docs/research-plan-20261004/任务一尺度验证与Qt5工作台结果.md) · [启动界面](D:/forAirsim/02_识别与回放工作台.cmd)。

**最新推进（2026-10-06，单机运动尺度独立重复）**：YOLO＋VLX链路下，1次20秒先导及3次同版本独立重复共200/200主帧正确、0错选，199次非零目标请求，端点净位移62.887～64.098毫米；全部安全收尾，零输出复测通过。修复50槽/41槽预声明不一致，原起飞前失败保留；353项相关测试通过。整帧完成198/200次晚到及清零后残余运动仍需改进，正式四方法与YOLO/VLX消融待采。第二、三章与中文图表已同步。[本轮结果与下一步](D:/forAirsim/docs/research-plan-20261004/任务一运动尺度重复验证与论文更新.md)。

**此前推进（2026-10-05，后台复核交接、恢复调度修复与中文图表）**：新增三次V5及一次V5P1实际飞行，全部安全收尾并关闭。后台确认新链后交接；活动指令下0.672/0.656秒停顿均保存15次查询及停顿内显式清零。原V5恢复调度错误完整保留，V5P1新会话在遮挡前重获跟踪，返回约4.032秒完成该帧处理（含执行与停稳）；40/41主测量帧中20正确、20无输出、0错选，13次非零请求均匹配目标。采样航程仅0.006068米，正式方法收益仍待测。299项软件检查通过，新图中文标注，第二、三章同步更新。下一步运动尺度验证，再补慢回复/属性误判与正式四方法、YOLO/VLX消融。[结果与下一步](D:/forAirsim/docs/research-plan-20261004/任务一后台复核交接与中文图表结果.md)。

**此前推进（2026-10-05，逐帧语义复核与源属性拒识）**：新增3次实际飞行并安全关闭；另1次UE启动绑定失败，运行器未启动，飞行数为零，单独保留。V3短飞行独立发现6帧错选、4次错误目标非零请求，原运行器PASS仍保留；V4采用前加入源RGB白横带检查，主测量39/41帧中18正确、21无输出、0错选，15次非零请求均匹配目标。V4实际采样航程仅0.00753米，尚不构成轨迹优化效果；两处停顿没有活动非零租约。253项相关测试通过，第二、三章补方法、失败与限制。下一步先改善后台复核交接，再验证带动作阻塞、慢回复和运动尺度，之后做四方法/消融正式重复。[本轮结果与下一步](D:/forAirsim/docs/research-plan-20261004/任务一逐帧语义复核与属性拒识结果.md) · [独立分析](D:/forAirsim/logs/t1_chain_analysis_final_v4p1_20261005/summary.md)。

**此前推进（2026-10-05，感知与自身控制解耦）**：新增独立控制线程并保存4次实际飞行，全部安全关闭，整体联合验证均FAIL。修正传输配置后三次零保持试次的保存自状态间隔最大78/94/78 ms；两处0.65 s阻塞期间仍保存15/14个状态读回。新批次没有正确身份输出或非零目标指令，不能替代此前厘米级闭环。另3次VLX组件请求均在约3.06–3.09 s读超时；146项相关测试通过。下一步定位语义时效与初始身份获取，再复验非零异步闭环，之后开展四方法和消融。[本轮记录与失败](D:/forAirsim/docs/research-plan-20261004/任务一异步调度与语义时效开发结果.md)。

**此前复核（2026-10-05，机载身份—深度—短XY闭环）**：保留5次开发飞行（含3次失败），全部通过原始或补充地面证明安全关闭。最新短飞行保存40/41时隙，24个当前选择经独立分割核对正确并发出24次非零目标指令；遮挡返回后连续14个正确输出，覆盖保存读回5.282秒。实际采样XY航程仅0.02664米，属于厘米级固定高度接口先导。整帧完成晚24次、实际动作门控决策晚11次，稳定0.4秒整流程仍待完成；下一步解耦自身状态控制与阻塞感知，再做四方法/消融。103项相关测试通过，旧V6.2十二个源码未改。[机载结果、失败与论文更新](docs/research-plan-20261004/任务一机载身份深度与低速轨迹集成结果.md) · [独立复核](logs/t1_identity_flight_analysis_final_20261005/summary.md)。

**此前推进（2026-10-04，持续身份与实时性）**：专用YOLO接入低频VLX、两帧确认、失锁与属性恢复。四版地面开发共16个自有UE启动、48次云请求，全部安全关闭；0.2 s时序失败保留，最新预登记标称0.4 s版实采123/124时隙、115个结果按时。两个白条目标会话均观察到恢复，同外观失锁后的返回阶段26个已采帧拒识；所选54个框均匹配目标，不代表全程身份正确率。144项相关测试通过，此前阶段新增飞行为零。已补第二、三章草稿，后续机载闭环见上。[此前实际结果](docs/research-plan-20261004/任务一持续身份与实时性开发结果.md) · [第二章](docs/thesis/第二章_问题建模与仿真平台.md) · [第三章](docs/thesis/第三章_单机感知轨迹优化.md)。

**此前推进（2026-10-04，专用气球 YOLO）**：八个独立 UE 会话按 4/2/2 分组，192 张图、224 个可见框、32 张负例；完成 GPU 20 epoch 训练和验证选模。两个新测试会话的 56 个实例全部检出，16 个双球画面均检出两球，8 张负例无误检；原 COCO 映射检出 25/56。71 项相关测试通过，采集与训练失败保留，新增飞行/云请求零。结果限于同地图/网格/红材质的检测开发，后续地面持续身份见上、轨迹实验待完成。[实际结果与后续闭环](docs/research-plan-20261004/任务一专用YOLO首轮结果与后续闭环.md)。

**此前推进（2026-10-04，任务一 YOLO/VLX）**：新增异步语义复核、较新帧本地关联、姿态/三维表面估计及独立身份评分。修正版地面相机会话九条件、十八基础帧和三十五实时帧通过场景验收；七条件有效语义关联、两条件拒识，无飞行。导出十八个气球框及两张纯负例；相关测试97项通过。旧场景问题及 API 失败分别保留，此阶段尚未训练专用 YOLO；后续首轮训练见上。[此前结果](docs/research-plan-20261004/任务一YOLO_VLX与评分验收结果.md) · [第三章内容与完成标准](docs/research-plan-20261004/任务一论文内容与完成标准.md)。

**当前研究主线（2026-09-30 开题更新）**：单机单目标感知任务轨迹优化 → 双机单目标互补感知与轨迹优化 → 两机对单气球的连续刷新任务分配与快速接触规划。本阶段仅做 AirSim 仿真。以 [`AirSim三项研究执行路线`](docs/research-plan-20260930/AirSim三项研究执行路线.md) 和 [`仿真任务登记`](configs/thesis_simulation_tasks.json) 为下一步入口。以下已有飞行属于平台和基线先导，尚未构成新轨迹优化方法的正式结果。

**最新推进（2026-10-03，独立制动测量）**：58项相关测试通过；原h=0.5 s模型在0.06 m/s四方向的采样停距上不足，失败保留，旧反序会话未开始。h=1.0 s候选在两个新UE会话、十六个0.03/0.06 m/s XY脉冲中验证，1614条请求和1630条查询核验通过。整个制动开发共十三个自有UE启动，十一次确认飞行均落地释放，两次确认未起飞的尝试单列；各版不混算方法成功率。仅支持已测低速、固定高度、自状态流程，原V6.2源码与配置保留。后续继续更高幅度/Z响应、相机姿态/三维与身份可见性评分，正式方法数据待采。[实测与失败](docs/research-plan-20261003/独立制动与响应实测结果.md) · [独立验证及图表](logs/braking_validation_v1p4_20261003/summary.md) · [硕士论文补齐路线](docs/research-plan-20261003/硕士论文成熟度与证据补齐路线.md)。

**V6.2开发重复（2026-10-03）**：时间索引参考、逐请求时效/变化率核验已实现；将居中代价与 RGB 检测质心视线对齐，加入采集后状态保护、独立空间预算检查及起飞前源码冻结核验。177项相关测试、2次独立短验收及单独正偏移2次验收通过。随后三工况、两种顺序共12次开发重复全部达标并落地释放，6对起点匹配，已记录运动和空间样本无超限；3918条目标请求核验通过。各版本分别登记。仍为单机单目标固定高度开发先导，完整3D、身份/遮挡与正式效果数据待后续准入。[V6.2实测记录](docs/research-plan-20261003/V6.2居中参考与空间门槛实测结果.md) · [完整批次与图表](logs/trajectory_matched_sweep_20261003_184327_9519224/analysis/summary.md) · [正式采数准入路线](docs/research-plan-20261003/T1正式采数准入与后续路线.md)。

V6原12次完整计划因请求变化间隔问题未启动；修正后的V6.1独立完成12次，效率6/6、感知4/6达标，12次落地释放、6对实测起点匹配。空间复核发现正偏移感知的两个末帧进入内侧保护界，分别作为任务失败和空间验收失败保留；速度/加速度未超限不能替代空间结论。[V6时间索引诊断](docs/research-plan-20261003/V6时间索引与时效调度实测.md) · [V6.1完整结果与空间复核](docs/research-plan-20261003/V6.1实测起点反序与侧边界诊断.md)。

此前V5.1修正支持框锚点后146项回归与2次独立短验收通过；正偏移2次完整飞行中感知完成、效率触发第七次减速后的次数保护，均安全收尾，104个规划输入支持框重投影通过。[V5.1结果](docs/research-plan-20261003/V5.1表面点锚点修正实测结果.md) · [当时飞行与图表](logs/trajectory_matched_sweep_20261003_165257_8024695/analysis_v2/summary.md)。

此前V5短验收2次、三工况完整先导6次（5次任务达标、6次安全收尾）分别保存。审计发现该版支持框锚点错误，原框角约束不能作为正确观测框或方法效果证据；实际飞行计数、失败和运动记录保留，V5与V5.1不混算成功率。[V5记录与投影更正](docs/research-plan-20261003/V5执行余量与观测重规划验收.md) · [锚点离线重放](logs/trajectory_matched_sweep_20261003_162504_2587397/footprint_anchor_replay/replay_receipt.json)。

此前V4的13次确认已开始尝试（12份终态、1次中断缺终态）保留独立记录；7次任务达标、12次落地释放。V4与V5源码和配置不同，不合并计算方法成功率。[V4结果及中断核对](docs/research-plan-20261003/实测起点匹配与偏移先导结果.md) · [此前8次接口记录](docs/research-plan-20261003/单机单目标推进记录与下一步.md)。

2026-09-30离线开发：候选轨迹规划器和气球刷新轮次协议的13项几何接口检查、3项同输入对照以及三策略事件重放通过；当时新模块与相关回归共70项单元测试通过。[几何验证](logs/trajectory_geometry_20260930_162648_714896/summary.md) · [刷新协议验证](logs/refreshed_target_protocol_20260930_162900_640240/summary.md)。

**基线先导（2026-09-30）**：单机第一阶段已完成统一 0.15 m/s 接近指令上限的重复性先导：3 次独立 Unreal 重启、12 次飞行、6 对完整对照；坐标与视觉策略各 6 次达到观察目标并安全收尾。已有失败和中断另行保留。该旧批次入口已从根目录移除；当前入口见 `命令使用说明.md`。结果见 [`先导汇总`](logs/phase1_pilot_20260930_104530_5529764/analysis/summary.md)，下一步与论文工期见 [`阶段进度与论文工期`](docs/阶段进度与论文工期_20260930.md)。

云端 VLX API 已在保存的 30 帧海事图像上完成图像＋文字回放，30 帧的返回框均与指定目标关联。这些帧来自一次悬停会话；现有云端请求未上传外部候选框，也未形成独立人工真值，不能据此声称 YOLO→VLX 的效果或泛化准确率。第一阶段证据与下一轮协议见 [`docs/第一阶段证据盘点.md`](docs/第一阶段证据盘点.md) 和 [`docs/毕业论文第一阶段实验协议.md`](docs/毕业论文第一阶段实验协议.md)。

随后另完成目标左右偏移的 12 次单机飞行与 374 张原始 RGB；位置先导结果见 [`变化条件汇总`](logs/phase1_lateral_sweep_20260930_112257_1344693/analysis/summary.md)。已在六张原图实测颜色、YOLO 与云端 VLX，接入分工及耗时见 [`YOLO与VLX接入决策`](docs/YOLO与VLX接入决策_20260930.md)。

---

For the first-stage **single-drone/single-target** comparison (DroneA only,
fixed-waypoint versus RGB/depth visual approach), see
`docs/单机单目标实验_第一阶段.md` and run
`& .\.venv\python.exe .\python\run_single_batch.py --repeats 1` after scene setup.

The single-drone maritime-map perception extension uses the saved harbor map,
three balloon instances and six scene conditions. See
`docs/单机海事地图目标判别实验.md` and run
`& .\.venv\python.exe .\python\maritime_single_experiment.py` after starting
the maritime scene.

## 🎯 VLX-Seek 视觉语言模型集成

基于 Om AI Lab 的 VLX-Seek API，实现了自然语言引导的细粒度目标识别。

### 快速开始

```powershell
# 1. 配置 API 凭证（编辑 .env 文件）
OM_API_KEY=你的密钥
OM_API_ENDPOINT=你的端点

# 2. 运行交互式演示
02_识别与回放工作台.cmd
```

### 核心功能

- ✅ **自然语言描述**：`"the rightmost red balloon"`
- ✅ **细粒度推理**：区分相似目标、理解空间关系
- ✅ **遮挡鲁棒**：部分遮挡场景下仍能准确识别
- ✅ **交互式界面**：实时绘制候选框、输入描述、查看结果

### 探索性对照（一次悬停会话的 30 帧）

| 方法 | 与指定目标关联的帧数 | 遮挡条件 | 相似干扰 |
|------|--------|---------|---------|
| YOLO 最高置信 | 16/30 | 0/5 | 1/5 |
| 最右规则 | 25/30 | 0/5 | 5/5 |
| **VLX-Seek 图像＋文字 API** | **30/30** | **5/5** | **5/5** |

详见：[`docs/VLX-API集成完成总结.md`](docs/VLX-API集成完成总结.md)

---

The saved maritime frames can be replayed through YOLO proposals and an optional
VLX-Seek region-grounding adapter. See `docs/VLX-Seek接入预检与海事回放.md` and
`python/vlx_seek_probe.py`. The local 8 GB GPU has not run the 10B VLX model.
The layout pilot in `configs/maritime_single_layout_pilot.json` swaps the target
between left and right in near and wide red-balloon pairs, with task descriptions
recorded before capture.
Three separate Unreal-process data sessions, AI-reviewed visible-box suggestions,
and exploratory YOLO/color comparisons are described in
`docs/海事左右布局探索性对照与VLX实跑状态.md`. Independent human labels and
candidate-bound VLX inference remain open; the cloud image-plus-text replay is
recorded separately in `logs/vlx_api_full_test.json`.

This project uses Unreal Engine 4.27.2, Microsoft AirSim and SimpleFlight to simulate two multirotors. `DroneA` performs the first approach; `DroneB` waits for a fallback command when the first attempt misses.

## Current status

- AirSim and the Blocks environment build successfully.
- Two vehicles are configured in `configs/settings.json`.
- The active settings file is installed at `%USERPROFILE%\Documents\AirSim\settings.json`.
- The two-drone smoke test has passed: takeoff, separated movement, hover, descent and disarm.
- Runtime logs are written to `logs/` and are ignored by Git.
- Live RPC verification confirms both DroneA and DroneB; a red balloon and string can be recreated with `scripts/start-scene.ps1`.
- Both vehicles explicitly configure named IMU, barometer, GPS, magnetometer,
  forward-distance and downward-distance sensors. Missions record advisory 1 Hz
  health snapshots without using them to change flight commands yet.

## Start the scene

From the project root in PowerShell:

```powershell
& .\scripts\start-scene.ps1
```

This connects to an existing simulation or launches Blocks, verifies both vehicles,
and creates the balloon configured in `configs/balloon.json`. Each run produces
`logs/scene_<timestamp>/scene_report.json`, `scene.log`, and `overview.png`.
Balloon objects are runtime objects; rerun after restarting the level. See
`docs/development-log.md` for validation results and remaining work.

## Project layout

- `python/`: Python control and coordination code
- `configs/`: versioned configuration templates
- `scripts/`: PowerShell and native build helpers
- `logs/`: local runtime logs, not committed
- `docs/`: installation and environment records
- `external/AirSim/`: upstream AirSim checkout and build tree

## Staging and approach demo

With the scene running, execute `& .\scripts\run-approach-demo.ps1`.
Both drones stage in the air; A approaches to about 2 metres from the balloon
centre while B holds. Press M in Blocks for the fixed overview. The demo ends
with simulation physics paused for inspection, not landed; rerun to replay.
Flight reports, telemetry and screenshots are saved in `logs/approach_<timestamp>/`.

## Full balloon demonstration (historical)

The retired `Run-Balloon-Demo.cmd` offered option 1 for A contact, or 2 for an intentional
A miss followed by B contact. The demonstration restores the balloon, verifies
fresh target collision evidence, removes the contacted balloon, returns both
drones to their pads and verifies ground contact before disarming. The scene
then pauses. See `docs/demo-guide.md` for Chinese instructions and limitations.

PowerShell: `& .\scripts\run-balloon-mission.ps1 -Scenario both`.
Logs and screenshots are in `logs/mission_<scenario>_<timestamp>/`;
`logs/experiment_summary.csv` includes passed and failed completed runs.

Choose **3** for the image-only balloon detection preview: both vehicles move to
observation positions, physics pauses, and a two-camera presence/absence comparison
opens. This mode leaves the drones airborne and paused. It does not use vision
to steer the drones. Results are saved under `logs/vision_<timestamp>/`.
Mission reports also include periodic annotated camera evidence while flying; image
detection is advisory and collision feedback decides a hit. A camera that is facing
away from the target can legitimately report no candidate.

Options 1 and 2 also open a live two-camera monitor for the entire mission. Closing
the monitor with Q/Esc does not stop the flight.
The monitor overlays the perception result: a green box/yellow center marks the
detected red balloon, and each panel displays `DETECTED` or `NO TARGET`.

The basic framework is now split into `configs/mission.json` (geometry and safety),
`python/mission_config.py` (validation), `python/target_provider.py` (ground-truth
and camera sensing ports), and `python/coordinator.py` (A-first/B-fallback
arbitration). The working mission runner remains the baseline while these modules
are integrated incrementally.

Choose **4** to open the live two-camera monitor (DroneA on the left, DroneB on the
right). Press Q or Esc in that window to close it.


### 2026-10-06 严格停稳与清零预算验证

两次V7P3独立UE会话均在冻结的60秒条件下达到20毫米门槛，净位移分别为22.972、23.468毫米。共保存300/300个主测量帧，独立评分300个正确、0个无输出、0个错选；实际非零目标派发145次，严格停稳、实际清零时效及派发预算均经独立核查。 [逐帧、指令及自身状态独立分析](D:/forAirsim/logs/t1_strict_settlement_analysis_final_20261006/report.json)

帧处理和控制停稳分别评价。原完整完成时间继续取两者最大值，仍与原0.4秒截止点比较。最新两次60秒运动共有30/300个帧处理晚到，原完整完成晚到172/300；分开计时不能将这些晚到改标为及时。控制停稳另与从声明租约结束起2秒的预先声明时限比较。

下一步在同一执行接口上补真实慢VLX回复、白条高光干扰和遮挡返回，继续保留所有失锁、无输出与晚到；随后单独冻结50/100毫米尺度及完整几何执行协议，统一固定跟随、距离规划、可见性规划和身份复核联合规划，开展YOLO检测后端及VLX开关消融。正式场景、参数、种子、全启动回合指标和独立重复另行冻结。任务一正式四方法效能、A3完整几何及A7—A10仍待补，六章结构与任务二、三安排保持原计划。

[本轮完整记录](D:/forAirsim/docs/research-plan-20261004/任务一严格停稳与清零预算验证结果.md)


### 2026-10-06 源证据扰动验证

本轮实际启动5个AirSim飞行会话，保存主帧620/620个，初始化帧另计31个。主帧独立评分为正确561帧、无输出59帧、其他0帧。原完整完成晚到310/620，帧处理晚到44/620。每个会话分别完成新进程地面复核和自有UE关闭；帧具有时间关联，独立实验单位是会话。 [逐帧、媒体、源链与控制独立分析](D:/forAirsim/logs/t1_source_fault_analysis_final_v8_20261006/report.json)

扰动行为独立判定全部通过。运动试次仍按原20毫米净位移标准单独评价，未达标会话数为0；拒绝旧证据和恢复正确输出不能替代运动尺度验证。控制参数沿用V7P3，原0.4秒完整完成截止点保持。这些结果属于少量受控开发会话，正式四方法效能、完整几何、YOLO/VLX消融和正式独立重复仍待补。

下一步单独冻结50/100毫米运动尺度及完整几何执行；再以统一接口、场景、种子、回合判据开展固定跟随、距离规划、可见性规划、身份复核联合规划及YOLO/VLX消融。

[本轮完整记录](D:/forAirsim/docs/research-plan-20261004/任务一源证据扰动验证结果.md)

### 2026-10-07 有界后台记录与尺度复验

本轮保留6次UE会话尝试、确认起飞5次；各次新鲜落地与所属UE关闭均已核实。旧V9五次尝试和212帧未评分中断流水继续保留。两次独立50毫米通过；100毫米因反馈自状态派发前时效失败提前停止，尚未通过。 [全部试次与独立核查](D:/forAirsim/logs/t1_pipeline_campaign_analysis_20261007/report.json)

| 会话 | 完整保存/开始/计划主帧 | 正确/无输出/其他或未知 | 净XY位移（毫米） | 独立判定 | 完整/处理晚到 |
|---|---:|---:|---:|---|---:|
| zero_v10 | 0/0/20 | 0/0/0 | 未测量 | 未通过 | 0/0 |
| zero_v10p1 | 20/20/20 | 20/0/0 | 0.000 | 通过 | 1/0 |
| 50_v10p1 | 0/0/400 | 0/0/0 | 未测量 | 未通过 | 0/0 |
| 50_v10p1_retry1 | 400/400/400 | 378/22/0 | 58.589 | 通过 | 250/111 |
| 50_v10p1_rep1 | 400/400/400 | 394/6/0 | 61.209 | 通过 | 245/86 |
| 100_v10p1 | 315/316/800 | 306/9/1 | 未测量 | 未通过 | 187/57 |

[本轮结果与后续](D:/forAirsim/docs/research-plan-20261004/任务一有界后台记录与尺度复验结果.md)

### 2026-10-07 反馈派发计时与末尾实际清零复验

V10P2新增两次安全收尾会话。零门控20/20主帧通过；50毫米先导400/400主帧全部正确、净XY位移62.614毫米，但最后一帧实际清零超源观测截止20毫秒，独立总核查未通过。本轮重复和100毫米未放行，旧V10P1两次50毫米通过及全部旧失败保留。[反馈时效改进与末尾清零复验结果](D:/forAirsim/docs/research-plan-20261004/任务一反馈时效改进与末尾清零复验结果.md)

467项完整回归通过；Qt修改后17项验证通过。Qt回放13次尝试，显示新版、反馈拒绝/恢复次数及清零超截止原因；中文时序图已保存。进程隔离仍为下一步计划，后续继续三维执行和四方法及YOLO/VLX消融。


### 2026-10-08 独立控制进程与采集中断复验

V11已实飞三次：零门控20/20通过；50毫米先导400/400完成、380正确/20无输出、净XY位移58.676毫米，通过独立开发核查；独立重复389/400完成、390已开始，因配对采集中断未通过，端点未知，100毫米未放行。三次停止、落地、释放、暂停与所属UE关闭已核实。 [本轮结果与证据](D:/forAirsim/docs/research-plan-20261004/任务一控制进程与采集中断复验结果.md)

重复中首次目标清零提前原截止184毫秒；原晚到提示来自旧收尾回执覆盖首次清零时钟。补充核查保留原报告和未通过判定。新采集诊断与首次清零修正各组12项测试通过，尚未集成实飞。Qt回放全部16次尝试、显示未知输出和两种时钟，20项界面测试与10张PNG检查通过，原生桌面交互未验收。 [Qt5指南](D:/forAirsim/docs/任务一Qt5实验工作台.md) · [下一轮接续](D:/forAirsim/docs/research-plan-20261004/任务一V11P1采集与首次清零接续计划.md)

当前仍是固定高度XY工程验证；正式三维执行、四方法对照及YOLO/VLX消融待完成。旧失败、无输出与212帧未评分流水继续保留。AI辅助实现和整理，解释以保存证据为准。

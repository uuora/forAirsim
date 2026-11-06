# 几何控制实验：跨电脑接续入口

当前工作范围是**一个高速三维连续过点任务**。控制方案选用几何期望姿态/推力外环加SimpleFlight姿态/角速度内环；不是完整自实现SE(3)控制器。下一步先验证连续过点控制，再比较Bézier、Hermite、B样条参考生成及时间分配；三方法比较尚未实现，不需要ABOM。

## 获取与打开

```powershell
git clone --branch codex/geometric-control-handoff-20261106 --recurse-submodules https://github.com/uuora/forAirsim.git
cd forAirsim
powershell -ExecutionPolicy Bypass -File scripts/setup-geometric-control.ps1
# 可选：记录本机UE路径，不改公共物理配置
powershell -ExecutionPolicy Bypass -File scripts/setup-geometric-control.ps1 -CheckOnly -EngineRoot 'C:/Epic/UE_4.27'
.\打开几何控制实验.cmd
```

需要Windows和Python3.10（`py -3.10`，或setup脚本`-Python`指定解释器）。安装脚本复用已有`.venv/python.exe`或标准`.venv/Scripts/python.exe`，不替换已有环境，不安装Torch，不启动UE或飞行。新窗口默认空闲；`powershell -File scripts/open-geometric-control-panel.ps1 -Check`仅做离线Qt检查。打开旧的轨迹面板仍进入旧系统。

## 当前实现与真实证据

- 核心：`python/geometric_validation/`；入口：`python/run_geometric_control_panel.py`和`python/run_geometric_control_validation.py`；独立配置：`configs/geometric_control*.json`。
- 世界NED、机体FRD；记录wxyz四元数。Python姿态wrapper的pitch/yaw反转已显式抵消。推力为归一化油门，名义映射不等于实测牛顿推力。
- 现有默认窗口仍是基础验证版。**默认waypoints中间参考速度为0，属于逐点停稳，不能称高速连续过点。** turn具有一个非零中间速度，但未形成完整高速连续任务覆盖。
- 8次尝试包含1次未解锁启动失败、7次实际仿真飞行；正常完成7、控制通过6、审计通过7、确认落地/解除解锁/释放API7。2倍速虽完成，参考SO(3)峰值16.72°、超10°约2.094秒，控制失败，已停止提速。
- 实际采样约32Hz，目标50Hz尚未达到。理想状态反馈、名义推力映射、未观测电机饱和、未建立源码到DLL的可复现构建证明，都是当前边界。
- `evidence.zip`约5.4MB，包含303个文件：原始状态、CSV、参考/指令、事件、报告、审计、源码快照、启动日志、Qt截图及图表；不丢弃失败。`evidence_manifest.json`逐文件绑定SHA256，省略的30个UE用户缓存文件有清单。原始内部绝对路径保持历史原样，不回填。

```powershell
# 只验包，不恢复、不连接AirSim
.\.venv\Scripts\python.exe scripts/restore-geometric-evidence.py
# 还原到指定空目录，或原路径恢复；拒绝覆盖不同内容
.\.venv\Scripts\python.exe scripts/restore-geometric-evidence.py --destination .
```

若复用conda布局，把上述解释器路径改为`.venv/python.exe`。恢复后结果入口为`reports/geometric_validation_summary_20261105_v2/SUMMARY.md`；跨电脑优先使用本目录`attempts.csv`的run ID，并在恢复后的reports树查找。历史说明中的D盘链接可能需按新根目录定位。审计器可对还原的单次目录重算，不调用AirSim。原始证据与未来新条件分组，不能将显示图或完成状态直接当控制通过。

## 仿真环境是另一层依赖

Git不携带UE安装、DLL、AirLib构建输出或虚拟环境。离线界面/测试可先开发；另一台机器运行飞行前需：

1. 完整初始化AirSim子模块，固定提交`d109f0dc25d4ae425eca7b279da33df5f5fcb42e`。按`patches/README.md`检查并应用`patches/airsim-local.patch`一次；`--reverse --check`通过表示已应用。不要覆盖另一台已有的不同本地补丁。
2. 按AirSim构建流程准备AirLib库，再以本机UE4.27及匹配的MSVC/SDK构建Blocks与AirSim插件。既有`build-airlib.cmd`和两个`build_dual_target_*.ps1`含原机器D盘/编译器路径，不能直接照搬；应改成本机路径后执行。源码/地图在补丁中，生成目录不进Git。
3. 保持`configs/dual_target_airsim_settings.json`原物理条件、`DynamicsProfile`与地图不变。设置本机`AIRSIM_UE_ROOT`或`.runtime/local_machine.json`。用`--check`核验文件；首次`--airsim --case hover --launch`会在解锁前核验实际设置、地图、DLL、未暂停物理与新鲜DTC_DYNAMICS回执，失败应保留。不要用stock SimpleFlight或缺失回执的DLL冒充当前条件。

## 后续优先任务

1. 新建连续过点任务配置及协议，明确各中间点非零参考速度、按顺序实际通过的区域/速度规则，终点静止；不要静默更改历史默认配置。
2. 补齐实际过点事件、各段时间和SO(3)/位置误差限值；控制/安全/审计分开，增益和门槛全方法共用。
3. 先排查实际32Hz时序与姿态动态，保守速度起步；上一速度级违约即停，不放宽阈值。
4. 控制器通过后建立三种具体参考生成方案，先检查换基等价性，再比较形状与时间分配。当前七次多项式验证器不等于已完成三种基线。

无障碍物、无视觉输入，capture_overview关闭。Qt图标仅显示实测偏航；截图不是AirSim相机数据。旧03入口、旧控制源码和实验完整保留。本次Git上传由用户明确授权；不自动生成导师汇报稿。

2026-11-06交接检查：独立检出39项测试通过，Qt空闲检查通过，303文件恢复校验，8条重审状态与原报告相同。本轮仅修复审计/恢复的Windows长路径兼容性，无新增飞行、阈值或控制律变化；未来源码指纹另组。

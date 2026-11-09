# 三种轨迹与 E0–E4 实测：当前接续入口

当前分支为 `codex/geometric-control-handoff-20261106`。已完成 Bézier、Hermite、B样条三种连续过点参考和五个无障碍、无视觉输入任务；minimum-snap 暂缓。统一采用几何期望姿态/推力外环加 SimpleFlight 内环，尚未实现实际闭环误差约束下的最快可行时间自动搜索。

## 获取与启动

```powershell
git clone --branch codex/geometric-control-handoff-20261106 --recurse-submodules https://github.com/uuora/forAirsim.git
cd forAirsim
powershell -ExecutionPolicy Bypass -File scripts/setup-geometric-control.ps1
.\打开几何控制实验.cmd
```

已有同分支检出可用 `git pull --ff-only`，有本地修改时先保留，勿强制覆盖。安装脚本补齐 Python 依赖并检查空闲界面，不安装 UE、不构建 DLL、不起飞；面板默认选一种环境和方案。UE4.27、AirLib、插件 DLL 与新鲜动力回执仍须另行准备，见[原始环境交接](../geometric_control_20261105/HANDOFF.md)。本机 `D:/foraisim` 入口只是转发器，新电脑使用本仓库根目录入口。

## 结果与证据恢复

本目录 `evidence.zip` 和 `evidence_manifest.json` 携带本轮已有日志、失败、冻结输入、源码及派生图，逐文件绑定 SHA256。原始几何控制调试包仍在 `experiments/geometric_control_20261105/`，两个包分别保留。

```powershell
# 只核验新包，不恢复、不连接或飞行
.\.venv\Scripts\python.exe scripts/restore-geometric-evidence.py --package experiments/geometric_planners_20261106
# 恢复到本仓库，已有不同内容会拒绝覆盖
.\.venv\Scripts\python.exe scripts/restore-geometric-evidence.py --package experiments/geometric_planners_20261106 --destination .
```

若 `.venv` 为 conda 布局，解释器改为 `.venv/python.exe`。恢复后直接在浏览器打开 `reports/geometric_environment_suite_20261106_160125_208546/index.html`。三维页离线可用，无 CDN 或服务器依赖；叠加/并排、方案勾选、参考开关、线宽、视角和比例均在图上操作。并排各图共用坐标范围并联动旋转/缩放，窄窗口竖排。静态 PNG/SVG、误差曲线与原始指标保留。

归档记录内原机器绝对路径保留历史原样，换电脑后不能直接据其绝对路径重新收集；现成结果页使用相对链接，交互数据已内嵌。需重算时另建明确路径映射，勿回写历史原始证据。恢复的页面不是新实验。

## 每次新测试后的对比入口（2026-11-09）

重开几何面板，点击底部“选择记录生成对比”，按环境勾选已有记录，再点“生成对比页”，默认自动打开。单次GUI记录和三方案批次记录均可使用；同组每种方案选一条，缺少方案也能生成。不同环境、源码或运行条件分别展示，失败记录保留；列表支持刷新、搜索和添加其他位置的单次记录目录。

每次结果保存在`reports/geometric_selected_comparison_时间戳/index.html`，保留现有离线交互三维图和每组完整曲线。生成时复制证据并独立重算，原始数据不改、不重新飞行；这个入口也能处理换电脑后恢复到本仓库的原始单次记录，不依赖清单中的旧机器绝对路径。新生成目录与11月6日的`evidence.zip`分别保留，本次没有重打包历史证据。

新增入口已用已有15条E0–E4记录核验，生成五组并确认原判定，原始输入未改；当前96项几何检查通过。Qt与桌面/窄屏离线浏览器核验在本机`reports/geometric_record_selection_20261109/`，没有新增飞行。

## 已完成与边界

- E0基准、E1直线、E2平面S弯、E3三维升降、E4大转向，各三种方案一次，实际15次；全部正常完成、跟踪/过点/独立审计/安全收尾通过，中途点39/39。首轮源码快照长路径失败及其他本机失败保留。
- 全程位置范数≤0.5m，参考SO(3)总姿态误差≤10°；中途点按顺序、参考时刻±1s、距点≤0.5m且实测速率≥0.5m/s。终点位置≤0.15m、速度≤0.15m/s、姿态误差≤5°持续1s。
- 五环境内部分别核对条件；每方法一次只证明开发跑通。Bézier/Hermite在当前端状态条件下等价换基，不推断算法优越性。SLSQP仅搜索参考运动学下的局部分段时长，不是全局或闭环时间最优。
- Qt按实测四元数显示三轴姿态，UE入口可显示/隐藏原生视口。图标为放大示意，显示与过点判据分开。关闭相机截图、无图像进入控制的约束继续保持。
- 11月6日几何模块84项离线测试通过；该批浏览器和原始输入不变核验在恢复后的总结果目录 `interactive_browser_check/verification.json`、`interactive_view_revision.json`。网页/Qt截图不是 AirSim 相机图。

## 后续工作

先完成闭环最快可行时间搜索的实验设计，再在固定位置/姿态门槛下逐级验证；不放宽门槛换速度。新源码与旧飞行分组，原始/失败记录保留。详细协议见 `docs/使用说明/几何控制验证_接口与协议.md`。发布本工作包由用户明确授权；未上传 UE、虚拟环境或本机 DLL。

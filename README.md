# AirSim 小场地轨迹实验

双击「打开轨迹实验面板.cmd」进入 8 × 8 × 2.5 m 小场地。路线可选共用航线、S 型、U 型回转和三维拱形。选好路线、曲线和计时后先点击「启动 / 连接 AirSim」，再点击「开始所选飞行」。面板可回放实测轨迹、打开原始数据和回看三航线对比。

- **三种普通基线**为分段五次 Hermite、分段五次 Bézier 和七次 B 样条。当前 Hermite 与 Bézier 使用等价的曲线表示。
- **TOPPQuad** 在选定几何上重计时。完整组使用原始解，自动调整组在原始解上搜索分段计时比例。两组使用同一个跟踪器和实测验收流程。
- **连续过点**从 (0, 0, 0.50) m 准备。三维参考速度和加速度上限分别为 1.60 m/s 与 1.50 m/s。协议 18 按目标到可见机身三维外框的最短距离验收，外框随实测姿态旋转。中间点按顺序要求距离 ≤ 0.30 m 且同一样本速度 ≥ 0.50 m/s。参考结束起，终点距离 ≤ 0.25 m 与速度 ≤ 0.50 m/s 须连续保持 1 s。姿态、边界和采样保护见小场地配置。关闭 AirSim 相机截图和自动录制。[小场地配置](configs/small_field.json)

## 近期实测

- [三航线配对回放](reports/small_field_retiming_pairs_20261126_111420_738367/index.html)对比三种基线各自的普通计时与自动调整。首轮 18 项中 17 项通过，三维拱形 Hermite 自动组的时序中止和通过的复测均保留。
- [三航线五方法对照](reports/small_field_three_routes_20261126_103941_029805/README.md)保留三条航线的普通基线、Hermite 完整组和自动调整组。完整组的三项规划预检拒绝均有回执。
- [三条航线的航点](configs/small_field_routes.json)取自冻结实验设计。仓库保留两批的冻结设计、逐架指标及审计回执。原始状态、指令、姿态和落地回执保存在实验机的 `logs/`。
- [协议 18 机身外框验收实测](reports/small_field_pilot_20261127_105616_772011/audit.json)和 [MINVO 离线预检对比](reports/small_field_minvo_20261127/comparison.json)分别记录外框冒烟及 Hermite 预检结果。

## 本机检查与复现

```powershell
.\打开轨迹实验面板.cmd --check
.\打开轨迹实验面板.cmd --route-preset s --verify-flights pass/hermite/ordinary pass/hermite/toppquad
```

第一条离屏检查面板，第二条执行两架次。路线参数还可用 `u-turn` 和 `arch`。本机复用 `.venv`、已编译的 AirSim Blocks 和 UE 4.27。TOPPQuad 使用 [lbTOPPQuad](https://github.com/maokat12/lbTOPPQuad) 的固定提交 `3db181cb0c76baaa1669f859f16a83060c2ee7a0`。首次配置时将该版本检出到 `.runtime/toppquad_upstream`。主控依赖见 `requirements.txt`，AirSim 本地修改见[补丁说明](patches/README.md)。

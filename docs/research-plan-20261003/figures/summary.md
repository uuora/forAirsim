# 轨迹接口短飞行诊断图

仅用于检查实际自状态采样与接口执行，不能作为感知轨迹优化的效果结论。

- 三组图分别使用主机单调时间、AirSim 查询时钟和无时间戳的记录索引；不同钟域不换算、不混接。每次飞行独立取时间零点。
- 保留不同阶段记录；查询时间可能相同，点数包含各阶段记录，不代表独立统计样本数。散点未插值，采样最大值不能证明连续过程峰值。
- 只从记录的实际三维速度、加速度计算范数；不使用规划参考、下发速度或命令变化率替代实际运动。
- world NED z 采用仿真世界原点、向下为正；它不是距地高度。初始起飞、返回、降落不在本图范围。
- V2 保护退出处未保留拒绝加速度向量；只显示保护原因，不补加速度点。V2 部分输入和命令起点缺独立时间戳，以记录索引呈现。
- executed_body_state 的最终查询覆写了状态字段；继承的末次子段主机时间未用于绘图，原字段仍保留在 CSV 中。
- actual_samples.csv 保存各采样及来源 JSON 路径；plot_points.csv 一行对应一个图点，同一采样可出现在两个独立钟域图中。provenance.json 保存原始报告、清单、批报告 SHA256 并核对上游哈希。

| 飞行 | 实际记录数 | z / a范数 / v范数点数 | 规划 / 完成命令 | 结束原因 |
|---|---:|---|---|---|
| V2 / efficiency | 17 | 17 / 14 / 16 | 2 / 2 | Measured initial acceleration exceeds planner budget |
| V2 / perception | 4 | 4 / 3 / 3 | 0 / 0 | Measured initial acceleration exceeds planner budget |
| V3 / efficiency | 40 | 35 / 40 / 40 | 3 / 3 | configured_interface_step_limit_reached |
| V3 / perception | 40 | 35 / 40 / 40 | 3 / 3 | configured_interface_step_limit_reached |

总实际记录 101 条，图点 437 个。所选飞行的 task_success 均按原始报告保留，PASS 不等同任务成功。

AI 辅助：此诊断脚本与图由 AI 辅助生成；原始飞行报告只读，未改写。

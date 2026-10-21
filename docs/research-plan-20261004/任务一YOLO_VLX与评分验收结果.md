# 任务一 YOLO、VLX 与评分验收结果

更新：2026-10-04。优先单机、一个指定气球实例、AirSim 仿真；相似气球为干扰物。当前已完成感知与评价接口的开发验收，正式任务一方法效果仍待采数。[任务登记](../../configs/thesis_simulation_tasks.json)、[完整归档](../../logs/t1_perception_closeout_v1p1_20261004/receipt.json)

后续更新：专用气球 YOLO 首轮采集、训练及独立来源测试已完成，详见 [首轮结果与后续闭环](任务一专用YOLO首轮结果与后续闭环.md)。本文以下保留本阶段的历史接口验收与未训练状态。

## 已完成的链路

YOLO 候选 → 带帧/图像哈希/任务版本的 VLX 异步复核 → 较新 YOLO 帧的本地唯一关联 → 配对深度中的实际像素表面估计 → 独立身份/可见性评分。VLX 不发送速度或航点，云请求未传输 YOLO proposals；不确定、过期、超时或歧义返回拒识。[有限感知策略](../../python/t1_semantic_perception.py)、[表面估计](../../python/t1_candidate_surface.py)、[评分内核](../../python/t1_identity_metrics.py)

三维观测采用 Scene 内参、图像响应相机四元数、车辆局部/世界平移和 DepthPerspective 射线距离；输出是代表可见表面点和 RGB 框中心视线，不是真实气球中心。身份正确和有效可见观察分别统计，全部预定时刻/已开始回合保留分母。[几何模块](../../python/t1_pose_geometry.py)、[实际几何核验](../../logs/t1_pose_probe_validation_v1p1_20261004/summary.json)

## 实际数据与失败

本轮四次自有 UE 启动，新增飞行零次，均有地面关闭证据。第一次因遮挡资产名在采集前失败；后两次旧场景虽然执行成功，但“移远”的干扰气球和遮挡物仍在视场，预期场景准入失败。旧原图和结果保留。修正版 V1.1 每帧检查未启用物体像素必须为零、完全遮挡时指定目标像素为零、应可见实例至少 500 像素。[逐次台账及纠正](../../logs/t1_perception_closeout_v1p1_20261004/receipt.json)、[V1.1 配置](../../configs/t1_perception_ground_probe_v1p1/protocol.json)

修正版一次 UE 会话包含九条件、十八基础帧及三十五实时帧。两机始终落地、控制释放；仅移动虚拟相机。该会话内条件和帧是嵌套观测，不能当作独立正式回合。[原始记录](../../logs/t1_ground_camera_campaign_20261004_003840_1015193/probe/report.json)、[进程关闭](../../logs/t1_ground_camera_campaign_20261004_003840_1015193/manifest.json)

| 接口/方法 | 修正版开发结果 | 暴露的问题 |
|---|---|---|
| 颜色最高形状评分 | 十八帧：十四指定、两错误干扰、两无输出 | 相似目标会选错 |
| YOLO 最高检测分数 | 十八帧：十四指定、四无输出 | 遮挡两帧、双气球两帧无候选 |
| 实时 YOLO＋VLX | 九条件：七个较新帧有效关联、两个不确定 | 无 YOLO 候选时拒绝选择 |
| 实时选择→表面估计 | 七个有效估计、两个保留无估计 | 估计几何通过不等于已验证定位精度 |

指定实例对应采用可见分割框 IoU≥0.5、相对干扰框差值≥0.05 的开发判据。真值只用于评分；表中结果不是 mAP、校准身份概率、持续身份保持或规划收益。[逐帧结果](../../logs/t1_ground_perception_replay_v1p1_20261004/report.json)、[实时独立核验](../../logs/t1_live_semantic_validation_v1p1_20261004/summary.json)、[表面估计核验](../../logs/t1_live_surface_validation_v1p1_20261004/summary.json)

新版实时核验中，当前帧年龄代理最大约 0.172 s，语义源帧年龄代理最大约 1.219 s；七次请求最小间隔 2.0 s。一百零五份实时媒体哈希及请求/当前帧绑定复算通过。年龄是读回接收代理，不能称曝光年龄。[时效记录](../../logs/t1_live_semantic_validation_v1p1_20261004/summary.json)

用户更新 API 后，旧保存图九次调用均正常返回，八个可见场景框对应指定气球、完全遮挡场景无输出，处理耗时中位数约 0.766 s；九次均超过 0.5 s。此批次复用旧场景，不混入新版干净场景结果。[更新后 API 回执](../../logs/t1_vlx_cloud_after_update_20261004/report.json)

新版 YOLO 处理耗时中位数约 0.062 s。当前单调时钟分辨率为 0.015625 s，颜色结果的零是计时量化；正式耗时使用高分辨率时钟，另记采集/排队成本。几何往返约 2.84e-14 px 只证明实现一致性，不能称标定或定位精度。[处理记录](../../logs/t1_ground_perception_replay_v1p1_20261004/report.json)、[计时说明](../../logs/t1_perception_closeout_v1p1_20261004/receipt.json)

## 标注、论文与下一步

已导出十八张 RGB、十八个可见 balloon 实例框和两张无可见气球负例。当前全部是一个会话的开发数据，未训练、未拆成独立训练/验证/测试；拆分检查拒绝同回合及相同图像跨组。YOLO 仍是 COCO `sports ball` 映射，尚未满足双气球工况。[开发集](../../datasets/t1_balloon_development_v1p1_20261004/manifest.json)、[导出与拆分检查](../../python/export_t1_balloon_dataset.py)

本轮相关测试九十七项通过，原 V6.2 十二个冻结模块保持不变。测试与仿真帧不作为正式回合数。[测试日志](../../logs/t1_final_related_tests_20261004.txt)、[源码核对](../../logs/t1_perception_closeout_v1p1_20261004/receipt.json)

第三章八节内容、变量、四基线、消融和图表已列成可执行清单。[任务一论文内容与完成标准](任务一论文内容与完成标准.md)、[正式协议草案](../../configs/t1_completion_protocol_v1.json)

接下来依次完成：独立回合气球数据及专用 YOLO → 指定实例时序关联/恢复与新版感知轨迹闭环 → 扩展运动范围验收 → 固定跟随、距离、可见性、身份联合四基线及检测/复核消融 → 第三章正式结果。现有虚拟相机验收不替代机体三维飞行，低速 XY 制动不外推到 Z/姿态控制。[运动范围待验证](../research-plan-20261003/下一轮三维与姿态接口验收计划.md)

实现依据：[Ultralytics 检测文档](https://docs.ultralytics.com/tasks/detect)、[VLX 推理说明](https://github.com/om-ai-lab/VLX-Seek/blob/main/docs/inference.md)、[AirSim 图像与分割说明](https://microsoft.github.io/AirSim/image_apis/)。AI 使用说明：代码、提纲和解释由 AI 辅助完成，实际结果保留媒体、哈希和核验；不替代导师或学校评审。

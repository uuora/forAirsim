# 任务一专用 YOLO 首轮结果与后续闭环

更新：2026-10-04。已完成按 UE 会话分组的首轮气球检测采集、训练和测试；任务一整体仍待持续身份与轨迹实验。[本轮归档](../../logs/t1_balloon_adaptation_closeout_v1p2_20261004/receipt.json)

## 数据与训练

采集前固定八个 UE 会话的种子和分组，改变球体尺度、相机距离/位置/姿态、目标位置及双球间距。每会话包含十二单球、八双球、四全遮挡负例，各一帧；共 192 张图、224 个可见框、32 张负例。所有实例都标为 class 0 balloon，身份标签不输入模型。[冻结计划](../../configs/t1_balloon_adaptation_v1p1_20261004/plan.json)、[数据清单](../../datasets/t1_balloon_adaptation_v1p1_20261004/manifest.json)

| 分组 | 独立 UE 会话 | 图像 | 可见气球框 | 负例 |
|---|---:|---:|---:|---:|
| 训练 | 4 | 96 | 112 | 16 |
| 验证 | 2 | 48 | 56 | 8 |
| 测试 | 2 | 48 | 56 | 8 |

整会话分组，相同 PNG 和完全相同的解码 RGB 不跨组，本批组内重复为零。近似画面和共同地图/材质仍存在，两个测试会话才是本批独立来源数量，四十八帧是会话内观测。[来源与重复核验](../../datasets/t1_balloon_adaptation_v1p1_20261004/manifest.json)

在 RTX 3070 Laptop GPU 上按预注册参数从原 YOLO11n 微调 20 个 epoch，验证集选 best.pt。训练仅使用九十六训练图和四十八验证图，测试图片/标签使用计数为零；不使用测试结果改阈值或再次选模。[训练回执](../../logs/t1_balloon_yolo_training_v1p2_20261004/receipt.json)

## 固定阈值的独立来源测试

同 CPU、640 输入、置信度阈值 0.25、NMS IoU 0.5；框以 IoU≥0.5 最大一对一匹配，重复框计误检。每方法四十八帧全部返回，检测耗时用预热后的 perf_counter_ns 测量，未含 AirSim 采集和 VLX 请求。[逐帧与分会话结果](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json)

| 指标 | 原 COCO sports ball 映射 | 专用 balloon |
|---|---:|---:|
| 正确检出实例 / 56 | 25/56 | 56/56 |
| 漏检实例 | 31 | 0 |
| 误检框 | 9 | 0 |
| 检测精确率 | 73.5% | 100.0% |
| 实例召回率 | 44.6% | 100.0% |
| 双球均检出 / 16 帧 | 5/16 | 16/16 |
| 有误检的负例 / 8 帧 | 0/8 | 0/8 |
| 检测中位耗时 | 49.63 ms | 48.61 ms |

专用模型另有低阈值 mAP 扫描：mAP50=0.995，mAP50–95≈0.890，记录见 [mAP 原始结果](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/specialist_map.json)。它与上表固定阈值的精确率/召回率不是同一计算口径；原 COCO 八十类未直接对单类标签计算 mAP。

可用于论文开发结果的 [对照图 PNG](../../logs/t1_balloon_adaptation_figures_v1p2_20261004/detector_comparison.png) 和 [矢量 PDF](../../logs/t1_balloon_adaptation_figures_v1p2_20261004/detector_comparison.pdf) 均从原始计数生成。另保留按清单固定选取的首个 [单球](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/first_test_1_visible_comparison.png)、[双球](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/first_test_2_visible_comparison.png) 和 [负例](../../logs/t1_balloon_yolo_evaluation_v1p2_20261004/first_test_0_visible_comparison.png) 对比图；首个双球例中原模型也检出两球，完整结果以上表及逐帧记录为准。

## 失败保留与代码接入

首版采集在第十九场景出现重建物体名称冲突，十八帧部分记录保留并排除。新入口直接更新已创建物体，全八会话均安全关闭。训练首试在第一 epoch 保存权重时遇到 Polars CPU 特征解析错误；重试仅使用标准 CSV 读取保存元数据，训练参数和数据保持冻结，失败快照均保留。[失败采集](../../logs/t1_balloon_adaptation_campaign_v1_20261004/ep01_train/manifest.json)、[失败训练](../../logs/t1_balloon_yolo_training_v1p1_20261004/receipt.json)、[重试适配说明](../../logs/t1_balloon_yolo_training_v1p2_20261004/receipt.json)

专用模型必须用 SpecialistBalloonDetector，严格检查模型名称为 {0: balloon}；旧 COCO 映射不会识别专用头。新检测器的候选格式已兼容 YoloVlxBackend(detector=...) 注入，VLX 继续负责低频语义复核。此处完成类别检测，尚未验证持续指定实例身份。[检测器](../../python/specialist_balloon_detector.py)、[接入配置](../../configs/t1_balloon_detector_v1p2.json)

本轮 71 项相关测试通过，原十二冻结飞行模块不变，新增飞行和云请求均为零。[测试日志](../../logs/t1_balloon_final_tests_20261004.txt)、[归档源码](../../logs/t1_balloon_adaptation_closeout_v1p2_20261004/receipt.json)

## 论文加入什么，下一步做什么

第三章感知实现与实验设置应加入：分组/种子/数据统计、可见掩膜监督、同类干扰气球、训练与验证选模、上述原模型对照、阴性误检、双球漏检、处理成本和失败案例。训练数据属于受控 AirSim 数据，不足以声称跨地图/外观泛化；这些检测结果不证明轨迹优化收益。[第三章内容与完成标准](任务一论文内容与完成标准.md)

接下来按顺序推进：

1. 接入专用候选，采集持续单目标序列，检查相似干扰、遮挡、重现、语义延迟；实现身份状态、唯一关联和恢复。
2. 将有效身份观测与表面估计接入新版轨迹闭环，完成运动范围和执行时效准入。
3. 固定跟随、距离、可见性、身份联合四基线使用相同检测后端；颜色检测/专用 YOLO 与 VLX 开关作为独立消融。
4. 冻结正式回合与全部失败分母，采集独立飞行重复后写第三章方法效果结果。

方法论文的主要证据仍是身份保持和轨迹优化实验。YOLO 单类检测分数不是校准身份概率；测试集已消费，后续改模型需另建未参与调参的测试版本。[正式协议草案](../../configs/t1_completion_protocol_v1.json)

实现参考：[Ultralytics 训练文档](https://docs.ultralytics.com/modes/train)、[验证文档](https://docs.ultralytics.com/modes/val)。AI 使用说明：代码、参数组织和结果说明由 AI 辅助，数值来自上述实际仿真/训练/测试回执，不代替导师和学校评审。

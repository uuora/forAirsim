# 初始曲线参数扰动：冻结实验结果

仅为离线点参考的几何与导数约束；按固定开发布局分层。

|曲线自由度|布局|搜索器|运行数|全压力条件通过|平均质量(低好)|平均参考时长(s)|平均搜索耗时(s)|
|---|---|---|---:|---:|---:|---:|---:|
|paired_bump|central|gaussian|20|20|0.405580|3.2645|2.1412|
|paired_bump|central|jde|20|20|0.416773|3.4888|1.9162|
|paired_bump|central|coordinate|20|20|0.399697|3.1870|2.1463|
|paired_bump|central|abom|20|20|0.430010|3.8389|2.0647|
|paired_bump|central|abom_perturb|20|20|0.409123|3.3272|2.2635|
|paired_bump|early_offset|gaussian|20|20|0.407703|3.2955|2.1028|
|paired_bump|early_offset|jde|20|20|0.415886|3.5015|1.8994|
|paired_bump|early_offset|coordinate|20|20|0.405347|3.2729|2.1092|
|paired_bump|early_offset|abom|20|20|0.431580|3.8869|1.8478|
|paired_bump|early_offset|abom_perturb|20|20|0.408205|3.3227|2.0852|
|paired_bump|two_obstacles|gaussian|20|20|0.413389|3.3372|2.7814|
|paired_bump|two_obstacles|jde|20|20|0.426408|3.5794|2.4100|
|paired_bump|two_obstacles|coordinate|20|20|0.417021|3.4048|2.5718|
|paired_bump|two_obstacles|abom|20|20|0.440473|3.9603|2.3054|
|paired_bump|two_obstacles|abom_perturb|20|20|0.415114|3.3698|2.6592|

压力条件均在训练声明的误差包络内；不构成未见布局或真实飞行的泛化证据。
配对区间按优化器种子计算，未校正多重比较；同一参考的条件不作为独立样本。
逐候选、冻结源码、选择和逐条件结果全部保留。AI协助实现、分析和写作。

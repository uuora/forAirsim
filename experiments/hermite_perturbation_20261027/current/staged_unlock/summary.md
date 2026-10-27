# 初始曲线参数扰动：冻结实验结果

仅为离线点参考的几何与导数约束；按固定开发布局分层。

|曲线自由度|布局|搜索器|运行数|全压力条件通过|平均质量(低好)|平均参考时长(s)|平均搜索耗时(s)|
|---|---|---|---:|---:|---:|---:|---:|
|paired_bump|central|staged_4to4|10|10|0.400413|3.2027|1.1009|
|paired_bump|central|staged_4to7|10|10|0.400228|3.1981|1.1264|
|paired_bump|central|staged_7to7|10|10|0.409585|3.3789|0.9802|
|paired_bump|early_offset|staged_4to4|10|10|0.400871|3.2290|1.0306|
|paired_bump|early_offset|staged_4to7|10|10|0.401184|3.2271|0.9991|
|paired_bump|early_offset|staged_7to7|10|10|0.410314|3.3633|0.9175|
|paired_bump|two_obstacles|staged_4to4|10|10|0.419379|3.4433|1.3421|
|paired_bump|two_obstacles|staged_4to7|10|10|0.420069|3.4441|1.3097|
|paired_bump|two_obstacles|staged_7to7|10|10|0.423491|3.5265|1.3839|

压力条件均在训练声明的误差包络内；不构成未见布局或真实飞行的泛化证据。
配对区间按优化器种子计算，未校正多重比较；同一参考的条件不作为独立样本。
逐候选、冻结源码、选择和逐条件结果全部保留。AI协助实现、分析和写作。

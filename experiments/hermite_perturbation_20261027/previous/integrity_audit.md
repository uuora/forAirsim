# Hermite 与扰动研究证据核查

总体状态：**PASS_WITH_LEGACY_UNBOUND**。

本核查只读保存的逐候选、选择、验证、摘要及源码快照，不重跑实验，不改写旧证据。

|目录|状态|训练运行|候选评价|验证条件评价|保留的失败条件|
|---|---|---:|---:|---:|---:|
|search_pilot|PASS_WITH_LEGACY_UNBOUND|20|2560|630|0|
|curve_search|PASS|80|10240|648|174|
|uncertainty_nominal|PASS|100|12800|6464|1587|
|uncertainty_margin|PASS|100|12800|6464|0|
|ellipsoid_nominal_final|PASS|100|12800|12928|1873|
|ellipsoid_isotropic|PASS|100|12800|12928|0|
|ellipsoid_directional|PASS|100|12800|12928|0|

覆盖：128 候选预算、逐 FE 顺序、历史前缀最小值、同种子共享初始化、已评价最优解、冻结边界、选择与训练绑定、验证参数与条件对应、失败保留、摘要及配对区间算术、源码快照 SHA-256。

search_pilot 的历史缺失绑定：training_manifest_binding × 20；validation_summary_manifest_binding × 1；validation_result_hash × 20；validation_manifest_binding × 20；baseline_manifest_binding × 1；baseline_selection_binding × 1。

`LEGACY_UNBOUND` 表示原文件没有相应哈希字段，不能补称已有绑定。其他可核查字段仍分别检查。

源码快照与 manifest 哈希相符，验证的是原记录对应的字节。后补恢复的历史快照不代表当时已经保存，哈希一致性也不能证明运行发生时间或训练/验证的真实先后顺序。

独立单位为优化器种子，同一参考的多个障碍条件不是独立飞行重复。失败计入全部条件汇总。本核查不证明算法优越性、物理安全、抗风或实飞效果。

AI 辅助编写和执行核查，详细检查计数、每个文件哈希及历史缺项见配套 JSON。

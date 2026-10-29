# 初始曲线参数扰动：独立记录审计

状态：**PASS**。

本审计只读取冻结源码和实验文件，不要求当前研究源码与冻结版本相同。

|项目|数量|
|---|---:|
|assertions|2990451|
|source_snapshots|9|
|training_runs|300|
|training_candidates|153600|
|training_infeasible_candidates|66161|
|stress_obstacles|256|
|stress_scenes|192|
|validation_references|303|
|validation_conditions|19392|
|failed_validation_conditions|0|
|summary_groups|15|
|paired_bootstrap_comparisons|12|

## 发现

未检出所覆盖范围内的不一致。

## 覆盖

- frozen source hashes; current source bytes not imported or required
- exact FE, generation indices, shared seeded initialization and anchor, parameter box
- evaluated incumbent, monotone prefix histories, all training check/score arithmetic
- independent full-curve jerk-energy and deformation-RMS arithmetic
- selection/training bindings, reconstructed within-ellipsoid stress scenes
- all validation result/check/certificate-status arithmetic and hash bindings
- summary full structure, group means/counts, seed-paired 10000-resample intervals

## 解释边界

- Record consistency does not prove record creation time, execution order, or that validation never influenced human decisions.
- Reported derivative hull and spatial certificate projection values are checked for internal arithmetic/status consistency, not independently rederived from every control hull.
- A stress result is hash-bound to the declared complete scene file; complete geometric oracles are not replayed by this auditor.
- A PASS is not a signed execution attestation, scientific novelty judgment, formal interval-arithmetic certificate, or aircraft/flight safety guarantee.

AI协助编写；审计结果由确定性程序计算。

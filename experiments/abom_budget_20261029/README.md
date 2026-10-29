# ABOM evaluation-budget experiments

Date: 2026-10-29. These are offline development experiments. No new AirSim or real flights were conducted. AI assisted implementation, analysis and writing.

The main method uses a valid objective lower bound to omit candidates that cannot enter stable elitist selection. It compares the bound with the **worst retained parent**, preserving the proposal stream, network targets and selected solution. Pruned scores and feasibility are recorded as unknown. The closed-form version uses the previously derived endpoint-preserving deformation energy for the declared default endpoint states.

|Campaign|Design|Evidence|
|---|---|---|
|`geometry_main`|180 original/screened pairs, 256 proposals each; numerical quality bound|[summary](geometry_main/summary.json), [audit](geometry_main/audit.json), [raw archive](geometry_main_raw.zip)|
|`geometry_closed_form`|Same 180 task/seed combinations, repeated paired timings with cheap closed-form bound|[summary](geometry_closed_form/summary.json), [audit](geometry_closed_form/audit.json), [raw archive](geometry_closed_form_raw.zip)|
|`control_cached_pilot`|9 original/fixed/adaptive triples, 64 proposals; 5D production planner parameters and fixed controller against a lagged point mass|[summary](control_cached_pilot/summary.json), [audit](control_cached_pilot/audit.json), [raw archive](control_cached_pilot_raw.zip)|
|`dropout_calibration`|180 unscreened searches, 256 evaluations; dropout .95/.5/.1 with common initialization|[paired comparisons](dropout_calibration/paired_comparisons.json), [audit](dropout_calibration/audit.json), [raw archive](dropout_calibration_raw.zip)|

Every raw archive includes complete per-candidate records and source snapshots. Compact manifests, run rows, summaries and audits are also unpacked. [Archive hashes](bundle_index.json) bind the packages. [Regression output](regression_tests.txt) records 195 relevant passing tests.

The two geometry timing campaigns use the same task/seed combinations; they are not 360 independent task repetitions. Each campaign audited 46,080 screened proposals and 8,208 pruned records. The reference and selected-score sequences were identical to their paired originals. Closed-form ABOM screening reduced complete evaluations by 24.92% and the six group median wall times by 18.74–22.96%. Other optimizers also benefit; this does not establish ABOM as the strongest optimizer.

The cached control pilot saved only two scenario rollouts across its screened arms. Adaptive ordering saved none. Its small timing differences do not establish a speedup. The existing model reports arrival/command checks, not synchronized reference tracking RMSE or motor saturation.

Dropout calibration improves some 4D conditions, with no universal setting across all tasks. These are fixed development layouts, not held-out generalization results. Choosing hyperparameters is baseline calibration, not a novel contribution.

The [excluded diagnostic archive](excluded_diagnostics.zip) preserves a smoke test affected by first-use optimizer loading and a stopped control pilot with unequal scenario caching. They are excluded from timing claims. Search time excludes one-time import/optimizer warmup, post-run validation and serialization; each manifest records the timing scope.

## Reproduce

Use Python with NumPy; ABOM runs additionally require PyTorch and the pinned upstream network. Fetch `xiaofangxd/ABOM` at commit `79a2baec700ce0abc6cf7d4cba1e897989d80516` into `tmp/abom_deploy_probe/upstream`, or pass `--upstream`. The loader checks network Git blob `5959aa7e4aef13c6b128e7c16e51d8a23205d030`. The upstream source is not redistributed here.

Run from the repository root, using new output directories:

```powershell
$env:PYTHONPATH='D:\forAirsim\python'
& '.\.venv\python.exe' python/run_screened_abom_research.py --output reports/repro_numerical
& '.\.venv\python.exe' python/run_screened_abom_research.py --bound-mode closed_form --output reports/repro_closed_form
& '.\.venv\python.exe' python/run_control_budget_research.py --seeds 301 302 303 --output reports/repro_control
& '.\.venv\python.exe' python/run_abom_dropout_research.py --output reports/repro_dropout
& '.\.venv\python.exe' python/audit_abom_budget_research.py reports/repro_numerical reports/repro_closed_form reports/repro_control reports/repro_dropout
& '.\.venv\python.exe' python/verify_abom_budget_bundles.py
```

For exact historical code use each archive's `source_snapshot`; audit scripts check snapshot hashes. The full Chinese [method and interpretation report](../../docs/research-hermite-20261027/ABOM计算预算改进_20261029.md) states the mathematical assumptions, negative results and next control-feedback hypothesis.

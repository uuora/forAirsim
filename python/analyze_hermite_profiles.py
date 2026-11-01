"""Paired optimizer-seed comparisons; obstacle conditions are never resampled as repeats."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/hermite_disturbance_20261027"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(before, after):
    a = read(BASE / before / "validation_summary.json")["rows"]
    b = read(BASE / after / "validation_summary.json")["rows"]
    assert read(BASE / before / "stress_cases.json")["cases"] == read(BASE / after / "stress_cases.json")["cases"]
    methods = sorted({r["method"] for r in a} - {"default"})
    rows = []
    for method in methods:
        left = {r["seed"]: r for r in a if r["method"] == method}
        right = {r["seed"]: r for r in b if r["method"] == method}
        assert left.keys() == right.keys()
        for seed in left:
            first = read(BASE / before / f"train_{method}_s{seed}.json")
            second = read(BASE / after / f"train_{method}_s{seed}.json")
            assert first["budget"] == second["budget"]
            assert first["initial_population_normalized"] == second["initial_population_normalized"]
        x = np.array([left[s]["reference_duration_s"] for s in sorted(left)])
        y = np.array([right[s]["reference_duration_s"] for s in sorted(left)])
        delta = y - x
        boot = np.random.default_rng(20261027).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        rows.append(dict(method=method, paired_optimizer_seeds=len(left),
                         fully_feasible_references_before=sum(r["feasible"] for r in left.values()),
                         fully_feasible_references_after=sum(r["feasible"] for r in right.values()),
                         mean_reference_duration_before_s=float(x.mean()), mean_reference_duration_after_s=float(y.mean()),
                         mean_delta_reference_duration_s=float(delta.mean()),
                         relative_change_percent=float((y.mean() / x.mean() - 1.) * 100),
                         exploratory_seed_bootstrap_95_percentile_ci_s=np.quantile(boot, [.025, .975]).tolist()))
    return dict(before=before, after=after, same_stress_geometries=True, same_initial_populations=True,
                candidates_budget_identical=True, methods=rows)


if __name__ == "__main__":
    comparisons = [compare("uncertainty_nominal", "uncertainty_margin"),
                   compare("ellipsoid_nominal_final", "ellipsoid_isotropic"),
                   compare("ellipsoid_isotropic", "ellipsoid_directional")]
    result = dict(scope="reference geometry and duration; no flight-time or physical robustness claim",
                  comparisons=comparisons, independent_unit="optimizer seed; all declared failures retained",
                  intervals="exploratory bootstrap; post-development comparisons, no multiplicity correction",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (BASE / "paired_profile_comparisons.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for comparison in comparisons:
        print(comparison["before"], "->", comparison["after"])
        for row in comparison["methods"]:
            print(row["method"], round(row["relative_change_percent"], 3), row["exploratory_seed_bootstrap_95_percentile_ci_s"])

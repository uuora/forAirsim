"""Independently audit a frozen initial-curve parameter-perturbation campaign.

Only JSON records, manifest declarations and hash-checked frozen source bytes
are consumed. Current trajectory_research modules are never imported. This
checks record consistency and recomputes scalar curve/summary arithmetic;
it does not establish when records were produced or physical flight safety.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


class InvalidArtifact(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object, parse_constant=reject)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b, *, atol=1e-11, rtol=1e-10):
    return finite(a) and finite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def key(family, layout, method, seed):
    return f"{family}__{layout}__{method}__s{seed}"


def tree_equal(actual, expected):
    """Require full structural coverage, allowing only scalar roundoff."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            tree_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            tree_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    return close(actual, expected)


class Auditor:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.findings = []
        self.counts = Counter()
        self.runs = {}
        self.choices = {}
        self.validations = {}
        self.initials = {}
        self.manifest = None
        self.boundary = None

    def require(self, condition, message):
        self.counts["assertions"] += 1
        if not condition:
            raise InvalidArtifact(message)

    def guard(self, context, function):
        try:
            return function()
        except (InvalidArtifact, KeyError, TypeError, ValueError, OSError,
                IndexError, ZeroDivisionError, OverflowError, StopIteration, AttributeError) as error:
            self.findings.append(dict(context=context, message=str(error)))
            return None

    def path(self, relative):
        self.require(isinstance(relative, str) and bool(relative), "nonempty relative path required")
        path = (self.output / relative).resolve()
        self.require(path.is_relative_to(self.output), "artifact path escapes campaign directory")
        return path

    def artifact(self, relative):
        return read(self.path(relative))

    def check_manifest(self):
        m = self.artifact("manifest.json")
        self.require(m["schema"] == "initial-curve-parameter-perturbation/1", "unsupported campaign schema")
        for name in ("families", "layouts", "methods", "seeds"):
            values = m[name]
            self.require(isinstance(values, list) and len(values) > 0 and
                         len(values) == len(set(values)), f"{name} must be nonempty and unique")
        self.require(all(type(s) is int and s >= 0 for s in m["seeds"]), "invalid optimizer seeds")
        self.require(set(m["families"]) <= {"single_bump", "paired_bump"}, "unsupported family")
        self.require(type(m["population"]) is int and m["population"] >= 4 and m["population"] % 2 == 0,
                     "invalid population")
        self.require(type(m["budget"]) is int and m["budget"] >= m["population"] and
                     m["budget"] % m["population"] == 0, "invalid exact FE budget")
        self.require(m["initialization"] in ("uniform", "lhs"), "unsupported initialization")
        self.require(m["certificate_degree"] == 7 and m["representation"] == "hermite",
                     "unexpected certificate degree or representation")
        for family in m["families"]:
            dim = 4 if family == "single_bump" else 7
            v = m["variables"][family]
            lo, hi, anchor = (np.array(v[x], dtype=float) for x in ("lower", "upper", "anchor"))
            self.require(lo.shape == hi.shape == anchor.shape == (dim,) and
                         np.isfinite([lo, hi, anchor]).all() and np.all(hi > lo) and
                         np.all(anchor >= lo) and np.all(anchor <= hi) and lo[-1] > 0,
                         f"invalid declared variables: {family}")
            self.require(len(v["names"]) == dim and len(set(v["names"])) == dim, "invalid parameter names")
        self.require(type(m["validation"]["count"]) is int and 1 <= m["validation"]["count"] <= 10000,
                     "invalid stress count")
        self.require(type(m["validation"]["seed"]) is int and m["validation"]["seed"] >= 0,
                     "invalid stress seed")
        layouts = {v["name"]: v["obstacles"] for v in m["task"]["layouts"]}
        self.require(len(layouts) == len(m["task"]["layouts"]) and set(m["layouts"]) <= set(layouts),
                     "layout declaration coverage differs")
        factor = np.array(m["task"]["geometry_factor_m"], dtype=float)
        self.require(factor.shape == (3, 3) and np.isfinite(factor).all(), "invalid ellipsoid factor")
        self.require(np.array(m["task"]["derivative_limits"]).shape == (3,) and
                     all(finite(x) and x > 0 for x in m["task"]["derivative_limits"]),
                     "invalid derivative limits")
        for relative, expected in m["source_sha256"].items():
            frozen = self.path("source_snapshot/" + relative)
            self.require(digest(frozen) == expected, f"frozen source hash mismatch: {relative}")
            self.counts["source_snapshots"] += 1
        self.require(bool(m["source_sha256"]), "no frozen sources")
        source = self.path("source_snapshot/python/trajectory_research/parameterizations.py")
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BoundaryStates")
        constants = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name) and n.target.id in ("p0", "p1", "v0", "v1", "a0", "a1")}
        self.require(set(constants) == {"p0", "p1", "v0", "v1", "a0", "a1"}, "frozen boundary state fields missing")
        self.boundary = {k: np.array(v, dtype=np.longdouble) for k, v in constants.items()}
        self.require(all(v.shape == (3,) and np.isfinite(v).all() for v in self.boundary.values()),
                     "invalid frozen physical boundary states")
        self.manifest, self.layouts = m, layouts
        self.manifest_sha = digest(self.output / "manifest.json")
        return m

    def independent_curve_values(self, parameters, family):
        """Direct power coefficients; no imported decoder or objective calls."""
        x = np.array(parameters, dtype=np.longdouble)
        duration, states = x[-1], self.boundary
        c = np.zeros((8, 3), dtype=np.longdouble)
        c[0] = states["p0"]
        c[1] = duration * states["v0"]
        c[2] = duration ** 2 * states["a0"] / 2
        dp = states["p1"] - c[0] - c[1] - c[2]
        dv = duration * states["v1"] - c[1] - 2 * c[2]
        da = duration ** 2 * states["a1"] - 2 * c[2]
        c[3] = 10 * dp - 4 * dv + da / 2
        c[4] = -15 * dp + 7 * dv - da
        c[5] = 6 * dp - 3 * dv + da / 2
        d = x[:3]
        e = x[3:6] if family == "paired_bump" else np.zeros(3, dtype=np.longdouble)
        bump = np.array([0, 0, 0, 64, -192, 192, -64, 0], dtype=np.longdouble)
        odd = np.array([0, 0, 0, -32, 160, -288, 224, -64], dtype=np.longdouble)
        c += bump[:, None] * d + odd[:, None] * e
        j = c[3:] * np.array([6, 24, 60, 120, 210], dtype=np.longdouble)[:, None]
        integrals = 1 / (np.arange(5)[:, None] + np.arange(5)[None, :] + np.longdouble(1))
        energy = sum(np.sum(coordinate[:, None] * coordinate[None, :] * integrals) for coordinate in j.T)
        energy /= duration ** 5
        rms = np.sqrt(np.longdouble(1024) / 3003 * (np.dot(d, d) + np.dot(e, e) / 60))
        return float(energy), float(rms)

    def check_details(self, details, parameters, family, layout, score, feasible, *, expected_energy=None):
        self.require(type(feasible) is bool and finite(score), "invalid score or feasible flag")
        checks = details["checks"]
        required = {"workspace", "speed", "acceleration", "jerk"} | {
            f"obstacle_{i}" for i in range(len(self.layouts[layout]))}
        self.require(isinstance(checks, dict) and set(checks) == required and
                     all(type(v) is bool for v in checks.values()), "check names/types differ from task")
        failures = sum(not v for v in checks.values())
        self.require(type(details["failed_checks"]) is int and type(details["total_checks"]) is int and
                     details["failed_checks"] == failures and details["total_checks"] == len(checks),
                     "failed/total check arithmetic differs")
        self.require(feasible == (failures == 0), "feasibility differs from check conjunction")
        bounds = details["derivative_bounds"]
        self.require(len(bounds) == 3 and all(finite(v) and v >= 0 for v in bounds), "invalid derivative bounds")
        for name, bound, limit in zip(("speed", "acceleration", "jerk"), bounds,
                                       self.manifest["task"]["derivative_limits"]):
            self.require(checks[name] == (bound <= limit), "derivative bound/check arithmetic differs")
        duration, energy, rms = (details[k] for k in ("duration_s", "jerk_energy_m2_s5", "deformation_rms_m"))
        self.require(close(duration, parameters[-1]) and duration > 0 and finite(energy) and energy >= 0
                     and finite(rms) and rms >= 0, "invalid physical quality inputs")
        actual_energy, actual_rms = expected_energy or self.independent_curve_values(parameters, family)
        self.require(close(energy, actual_energy, atol=2e-7, rtol=2e-8), "independent jerk energy differs")
        self.require(close(rms, actual_rms), "independent RMS deformation differs")
        quality = .9 * duration / (duration + 5.) + .05 * energy / (energy + 30.) + .05 * rms / (rms + 1.)
        self.require(close(details["quality_term"], quality) and 0 <= quality < 1,
                     "quality arithmetic differs")
        self.require(close(score, quality if feasible else 1 + failures / len(checks) + quality),
                     "feasibility-first score arithmetic differs")

    def training(self, combo):
        family, layout, method, seed = combo
        m = self.manifest
        relative = "train/" + key(*combo) + ".json"
        path, run = self.path(relative), self.artifact(relative)
        for name, expected in zip(("family", "layout", "method", "seed"), combo):
            self.require(run[name] == expected, f"training {name} binding differs")
        self.require(run["manifest_sha256"] == self.manifest_sha and run["budget"] == m["budget"]
                     and run["population"] == m["population"], "training manifest/budget binding differs")
        self.require(run["initialization"] == m["initialization"] and run["normalized_domain"] == [-1., 1.],
                     "initialization/domain declaration differs")
        variables = m["variables"][family]
        lo, hi, anchor = (np.array(variables[k]) for k in ("lower", "upper", "anchor"))
        population, dim = m["population"], len(lo)
        rng = np.random.default_rng(seed)
        initial = rng.uniform(-1, 1, (population, dim)) if m["initialization"] == "uniform" else np.stack(
            [(rng.permutation(population) + rng.random(population)) / population * 2 - 1 for _ in range(dim)], axis=1)
        initial[0] = 2 * (anchor - lo) / (hi - lo) - 1
        observed = np.array(run["initial_population_normalized"])
        self.require(observed.shape == initial.shape and np.array_equal(observed, initial),
                     "seed-reconstructed initial population differs")
        group = (family, layout, seed)
        if group in self.initials:
            self.require(np.array_equal(observed, self.initials[group]), "initial population differs across methods")
        else:
            self.initials[group] = observed
        records = run["evaluations"]
        self.require(len(records) == m["budget"], "wrong exact evaluation budget")
        values = np.array([r["parameters"] for r in records])
        self.require(values.shape == (m["budget"], dim) and np.isfinite(values).all() and
                     np.all(values >= lo) and np.all(values <= hi), "candidate parameters outside declared physical box")
        physical_initial = lo + (initial + 1) * .5 * (hi - lo)
        self.require(np.allclose(values[:population], physical_initial, rtol=0, atol=2e-14),
                     "initial evaluated candidates differ from common population")
        scores = []
        for i, row in enumerate(records):
            self.require(type(row["fe"]) is int and type(row["generation"]) is int and
                         row["fe"] == i + 1 and row["generation"] == i // population,
                         "nonconsecutive FE or generation")
            self.require(i >= population or row["source"] == "initial", "initial source label differs")
            self.require(finite(row["evaluation_s"]) and row["evaluation_s"] >= 0, "invalid evaluation duration")
            self.check_details(row["details"], row["parameters"], family, layout, row["score"], row["feasible"])
            scores.append(row["score"])
        self.require(run["best_score"] == min(scores) and any(row["parameters"] == run["best_parameters"]
                     and row["score"] == run["best_score"] for row in records), "incumbent is not an evaluated global minimum")
        history = run["history"]
        self.require([h["fe"] for h in history] == list(range(population, m["budget"] + 1, population)),
                     "history FE coverage differs")
        previous = math.inf
        for h in history:
            self.require(close(h["best"], min(scores[:h["fe"]])) and h["best"] <= previous,
                         "history is not monotone prefix minimum")
            previous = h["best"]
        self.require(finite(run["elapsed_s"]) and run["elapsed_s"] >= 0 and
                     close(run["evaluation_s"], sum(r["evaluation_s"] for r in records), atol=1e-7),
                     "search/evaluation timing arithmetic differs")
        if method.startswith("abom"):
            self.require(run["upstream_commit"] == "79a2baec700ce0abc6cf7d4cba1e897989d80516",
                         "ABOM upstream commit differs")
        self.runs[combo] = (relative, run, digest(path))
        self.counts["training_runs"] += 1
        self.counts["training_candidates"] += len(records)
        self.counts["training_infeasible_candidates"] += sum(not r["feasible"] for r in records)

    def selection(self):
        value = self.artifact("selection.json")
        self.require(value["manifest_sha256"] == self.manifest_sha and value["selection_basis"] == "training_only",
                     "selection manifest/basis differs")
        rows = value["incumbents"]
        actual = [(r["family"], r["layout"], r["method"], r["seed"]) for r in rows]
        self.require(len(actual) == len(self.expected) and set(actual) == set(self.expected),
                     "selection coverage differs from design")
        for combo, row in zip(actual, rows):
            relative, run, sha = self.runs[combo]
            self.require(row["training_file"] == relative and row["training_sha256"] == sha and
                         row["parameters"] == run["best_parameters"] and row["training_score"] == run["best_score"],
                         "selected candidate/training hash binding differs")
            self.choices[combo] = row["parameters"]
        self.selection_sha = digest(self.output / "selection.json")

    def stress(self):
        value = self.artifact("stress_cases.json")
        self.require(value["selection_sha256"] == self.selection_sha, "stress/selection binding differs")
        self.require(set(value["cases"]) == set(self.manifest["layouts"]), "stress layout coverage differs")
        factor = np.array(self.manifest["task"]["geometry_factor_m"])
        increment = self.manifest["task"]["radius_increment_m"]
        for layout_index, layout in enumerate(self.manifest["layouts"]):
            actual, nominal = value["cases"][layout], self.layouts[layout]
            self.require(len(actual) == self.manifest["validation"]["count"], "stress scene count differs")
            rng = np.random.default_rng(self.manifest["validation"]["seed"] + layout_index)
            for scene in actual:
                self.require(len(scene) == len(nominal), "stress obstacle count differs")
                for obstacle, base in zip(scene, nominal):
                    z = rng.normal(size=3)
                    z /= max(float(np.linalg.norm(z)), 1e-300)
                    z *= rng.random() ** (1 / 3.)
                    centre = np.array(base["centre"]) + factor @ z
                    radius = float(rng.uniform(max(float(np.nextafter(0., 1.)), base["radius"] - increment),
                                               base["radius"] + increment))
                    observed = np.array(obstacle["centre"])
                    delta = observed - np.array(base["centre"])
                    latent, *_ = np.linalg.lstsq(factor, delta, rcond=None)
                    self.require(observed.shape == (3,) and np.isfinite(observed).all() and
                                 np.allclose(factor @ latent, delta, rtol=0, atol=1e-12) and
                                 np.linalg.norm(latent) <= 1 + 1e-12, "stress centre outside ellipsoid")
                    self.require(finite(obstacle["radius"]) and obstacle["radius"] > 0 and
                                 abs(obstacle["radius"] - base["radius"]) <= increment + 1e-12,
                                 "stress radius outside declared interval")
                    self.require(np.allclose(observed, centre, rtol=0, atol=2e-14) and obstacle["radius"] == radius,
                                 "seed-reconstructed stress geometry differs")
                    self.counts["stress_obstacles"] += 1
                self.counts["stress_scenes"] += 1
        self.stress_sha = digest(self.output / "stress_cases.json")
        self.stress_cases = value["cases"]

    def certificate(self, certificate, certified):
        status = certificate["status"]
        self.require(status in ("CERTIFIED", "VIOLATED", "UNRESOLVED") and certified == (status == "CERTIFIED"),
                     "check/certificate status differs")
        witness, unresolved = certificate["violation_witness"], certificate["unresolved_intervals"]
        self.require((status == "VIOLATED") == (witness is not None), "violation status lacks/excludes witness")
        if status != "VIOLATED":
            self.require((status == "UNRESOLVED") == bool(unresolved), "unresolved status/intervals differ")
        if status == "CERTIFIED":
            self.require(certificate["certified_leaves"] > 0 and not unresolved and witness is None,
                         "certified interval has inconsistent leaf evidence")

    def evaluation_result(self, result, parameters, family, layout, robust, *, scene=None, expected_values=None):
        self.require(result["parameters"] == parameters and result["family"] == family and result["layout"] == layout
                     and result["representation"] == self.manifest["representation"] and result["robust"] is robust,
                     "objective result binding differs")
        details = result["score_details"]
        self.check_details(details, parameters, family, layout, result["score"], result["feasible"],
                           expected_energy=expected_values)
        self.certificate(result["workspace_certificate"], details["checks"]["workspace"])
        certificates = result["geometry_certificates"]
        self.require(len(certificates) == len(self.layouts[layout]), "geometry certificate count differs")
        for i, certificate in enumerate(certificates):
            self.certificate(certificate, details["checks"][f"obstacle_{i}"])
            if robust:
                self.require(certificate["uncertainty_factor"] == self.manifest["task"]["geometry_factor_m"],
                             "training certificate uncertainty factor differs")
                inflated = self.layouts[layout][i]["radius"] + self.manifest["task"]["radius_increment_m"] \
                    + self.manifest["task"]["clearance_m"]
                self.require(close(certificate["inflated_radius_without_centre_uncertainty"], inflated),
                             "training inflated radius arithmetic differs")
                leaves = certificate["leaf_certificates"]
                self.require(certificate["certified_leaves"] == len(leaves), "support-plane leaf count differs")
                for leaf in leaves:
                    normal = np.array(leaf["normal"])
                    support = float(np.linalg.norm(np.array(certificate["uncertainty_factor"]).T @ normal))
                    self.require(normal.shape == (3,) and np.isfinite(normal).all() and np.linalg.norm(normal) <= 1 + 1e-14,
                                 "invalid support-plane normal")
                    self.require(close(leaf["uncertainty_support_m"], support) and
                                 close(leaf["separating_margin_m"], leaf["projection_lower_bound_m"] - inflated - support)
                                 and leaf["separating_margin_m"] > certificate["numerical_margin"],
                                 "support-plane margin arithmetic differs")
                if certificate["status"] == "CERTIFIED":
                    intervals = sorted(leaf["interval"] for leaf in leaves)
                    self.require(close(intervals[0][0], 0) and close(intervals[-1][1], 1) and all(
                        0 <= start < end <= 1 for start, end in intervals) and all(
                        close(a[1], b[0]) for a, b in zip(intervals, intervals[1:])),
                        "certified support-plane leaves do not partition [0,1]")
            else:
                self.require(close(certificate["inflated_radius"], scene[i]["radius"] + self.manifest["task"]["clearance_m"]),
                             "stress certificate inflated radius differs")

    def validation(self, combo):
        family, layout, method, seed = combo
        parameters = self.manifest["variables"][family]["anchor"] if method == "initial" else self.choices[combo]
        relative = "validation/" + key(*combo) + ".json"
        value = self.artifact(relative)
        for field, expected in zip(("family", "layout", "method", "seed"), combo):
            self.require(value[field] == expected, f"validation {field} binding differs")
        self.require(value["parameters"] == parameters and value["manifest_sha256"] == self.manifest_sha and
                     value["selection_sha256"] == self.selection_sha and value["stress_sha256"] == self.stress_sha,
                     "validation candidate/artifact binding differs")
        curve_values = self.independent_curve_values(parameters, family)
        self.evaluation_result(value["training_check"], parameters, family, layout, True, expected_values=curve_values)
        if method != "initial":
            self.require(value["training_check"]["score"] == self.runs[combo][1]["best_score"],
                         "validation recomputed training score differs from selection")
        self.require(len(value["cases"]) == value["conditions"] == self.manifest["validation"]["count"],
                     "validation condition coverage differs")
        for result, scene in zip(value["cases"], self.stress_cases[layout]):
            self.evaluation_result(result, parameters, family, layout, False, scene=scene, expected_values=curve_values)
        failures = sum(not result["feasible"] for result in value["cases"])
        self.require(type(value["failures"]) is int and type(value["all_conditions_pass"]) is bool and
                     value["failures"] == failures and value["all_conditions_pass"] == (failures == 0),
                     "validation conjunction/failure arithmetic differs")
        details = value["training_check"]["score_details"]
        for field, original in (("quality", "quality_term"), ("duration_s", "duration_s"),
                                ("jerk_energy_m2_s5", "jerk_energy_m2_s5"), ("deformation_rms_m", "deformation_rms_m")):
            self.require(value[field] == details[original], "validation quality projection differs")
        self.validations[combo] = (relative, value, digest(self.path(relative)))
        self.counts["validation_references"] += 1
        self.counts["validation_conditions"] += len(value["cases"])
        self.counts["failed_validation_conditions"] += failures

    def summary(self):
        summary = self.artifact("validation_summary.json")
        for field, expected in (("manifest_sha256", self.manifest_sha), ("selection_sha256", self.selection_sha),
                                ("stress_sha256", self.stress_sha)):
            self.require(summary[field] == expected, "validation summary artifact binding differs")
        rows = summary["rows"]
        row_keys = [(r["family"], r["layout"], r["method"], r["seed"]) for r in rows]
        self.require(len(row_keys) == len(self.expected_validation) and set(row_keys) == set(self.expected_validation),
                     "validation summary coverage differs")
        columns = ("family", "layout", "method", "seed", "quality", "duration_s", "conditions", "failures",
                   "all_conditions_pass", "jerk_energy_m2_s5", "deformation_rms_m")
        for combo, row in zip(row_keys, rows):
            relative, result, sha = self.validations[combo]
            expected = {field: result[field] for field in columns}
            expected.update(result_file=relative, result_sha256=sha)
            self.require(tree_equal(row, expected), "validation summary row/value/hash differs")
        m = self.manifest
        groups, pairs = [], []
        for family, layout in itertools.product(m["families"], m["layouts"]):
            for method in m["methods"]:
                selected = [r for r in rows if (r["family"], r["layout"], r["method"]) == (family, layout, method)]
                runs = [self.runs[(family, layout, method, r["seed"])][1] for r in selected]
                groups.append(dict(family=family, layout=layout, method=method, runs=len(selected),
                    full_pass_runs=sum(r["all_conditions_pass"] for r in selected), failed_conditions=sum(r["failures"] for r in selected),
                    mean_quality=float(np.mean([r["quality"] for r in selected])),
                    mean_duration_s=float(np.mean([r["duration_s"] for r in selected])),
                    mean_jerk_energy_m2_s5=float(np.mean([r["jerk_energy_m2_s5"] for r in selected])),
                    mean_search_s=float(np.mean([r["elapsed_s"] for r in runs])),
                    mean_final_training_score=float(np.mean([r["best_score"] for r in runs]))))
            if "abom" in m["methods"]:
                for method in m["methods"]:
                    if method != "abom":
                        pairs.append(dict(family=family, layout=layout, first=method, second="abom", **self.paired(
                            {(r["seed"]): r for r in rows if (r["family"], r["layout"], r["method"]) == (family, layout, method)},
                            {(r["seed"]): r for r in rows if (r["family"], r["layout"], r["method"]) == (family, layout, "abom")})))
        if {"single_bump", "paired_bump"} <= set(m["families"]):
            for layout, method in itertools.product(m["layouts"], m["methods"]):
                pairs.append(dict(layout=layout, method=method, first="paired_bump", second="single_bump", **self.paired(
                    {r["seed"]: r for r in rows if (r["family"], r["layout"], r["method"]) == ("paired_bump", layout, method)},
                    {r["seed"]: r for r in rows if (r["family"], r["layout"], r["method"]) == ("single_bump", layout, method)})))
        expected = dict(schema=m["schema"], manifest_sha256=self.manifest_sha,
            validation_summary_sha256=digest(self.output / "validation_summary.json"), groups=groups, paired_comparisons=pairs,
            training_runs=len(self.expected), training_candidates=m["budget"] * len(self.expected),
            validation_conditions=sum(r["conditions"] for r in rows), failed_validation_conditions=sum(r["failures"] for r in rows),
            initial_rows=[r for r in rows if r["method"] == "initial"], scope=m["task"]["scope"])
        self.require(tree_equal(self.artifact("summary.json"), expected), "summary structure, counts, means or paired bootstrap differs")
        self.counts["summary_groups"] = len(groups)
        self.counts["paired_bootstrap_comparisons"] = len(pairs)

    @staticmethod
    def paired(first, second):
        seeds = sorted(first)
        delta = np.array([first[s]["quality"] - second[s]["quality"] for s in seeds])
        # Resample seed-paired differences, never correlated obstacle conditions.
        estimates = np.random.default_rng(20261027).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        return dict(field="quality", seeds=seeds, mean_delta=float(np.mean(delta)),
            exploratory_bootstrap_95_ci=np.quantile(estimates, [.025, .975]).tolist(),
            wins=int(np.count_nonzero(delta < -1e-12)), ties=int(np.count_nonzero(abs(delta) <= 1e-12)),
            losses=int(np.count_nonzero(delta > 1e-12)),
            interpretation="negative favors first; paired optimizer seeds within fixed layout; unadjusted for multiplicity")

    def run(self):
        if self.guard("manifest/source snapshots", self.check_manifest) is not None:
            m = self.manifest
            self.expected = list(itertools.product(m["families"], m["layouts"], m["methods"], m["seeds"]))
            for combo in self.expected:
                self.guard("train/" + key(*combo), lambda combo=combo: self.training(combo))
            if len(self.runs) == len(self.expected):
                self.guard("selection", self.selection)
                if hasattr(self, "selection_sha"):
                    self.guard("stress geometry", self.stress)
                    if hasattr(self, "stress_sha"):
                        self.expected_validation = self.expected + list(itertools.product(
                            m["families"], m["layouts"], ["initial"], [None]))
                        for combo in self.expected_validation:
                            self.guard("validation/" + key(*combo), lambda combo=combo: self.validation(combo))
                        if len(self.validations) == len(self.expected_validation):
                            self.guard("summary arithmetic", self.summary)
        return dict(schema="initial-curve-parameter-perturbation-audit/1",
            status="PASS" if not self.findings else "FAIL", output=str(self.output),
            auditor_sha256=digest(__file__), python=platform.python_version(), numpy=np.__version__,
            counts=dict(self.counts), findings=self.findings,
            coverage=["frozen source hashes; current source bytes not imported or required",
                "exact FE, generation indices, shared seeded initialization and anchor, parameter box",
                "evaluated incumbent, monotone prefix histories, all training check/score arithmetic",
                "independent full-curve jerk-energy and deformation-RMS arithmetic",
                "selection/training bindings, reconstructed within-ellipsoid stress scenes",
                "all validation result/check/certificate-status arithmetic and hash bindings",
                "summary full structure, group means/counts, seed-paired 10000-resample intervals"],
            limitations=["Record consistency does not prove record creation time, execution order, or that validation never influenced human decisions.",
                "Reported derivative hull and spatial certificate projection values are checked for internal arithmetic/status consistency, not independently rederived from every control hull.",
                "A stress result is hash-bound to the declared complete scene file; complete geometric oracles are not replayed by this auditor.",
                "A PASS is not a signed execution attestation, scientific novelty judgment, formal interval-arithmetic certificate, or aircraft/flight safety guarantee."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auditor = Auditor(args.output)
    report = auditor.run()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 初始曲线参数扰动：独立记录审计", "", f'状态：**{report["status"]}**。', "",
             "本审计只读取冻结源码和实验文件，不要求当前研究源码与冻结版本相同。", "", "|项目|数量|", "|---|---:|"]
    lines += [f"|{name}|{value}|" for name, value in report["counts"].items()]
    lines += ["", "## 发现", ""] + ([f'- {x["context"]}: {x["message"]}' for x in report["findings"]] or ["未检出所覆盖范围内的不一致。"])
    lines += ["", "## 覆盖", ""] + ["- " + value for value in report["coverage"]]
    lines += ["", "## 解释边界", ""] + ["- " + value for value in report["limitations"]]
    lines += ["", "AI协助编写；审计结果由确定性程序计算。", ""]
    (output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], counts=report["counts"], findings=report["findings"][:10]), ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

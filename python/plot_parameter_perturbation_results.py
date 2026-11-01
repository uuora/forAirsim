"""Create scientific plots and CSV tables from one frozen perturbation campaign.

Reference duration, scalar quality, and measured search wall time are distinct
quantities. Path examples use a declared method and the lower-median training
seed for each family; validation outcomes never select the displayed curves.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from run_parameter_perturbation_research import digest, load_selection, read
from trajectory_research.parameterizations import curve_from_parameters


METHOD_LABELS = {"random": "Random", "gaussian": "Gaussian", "de": "DE", "jde": "jDE",
                 "coordinate": "Coordinate", "abom": "ABOM", "abom_perturb": "ABOM + P"}
METHOD_COLORS = {"random": "#7F8C8D", "gaussian": "#0072B2", "de": "#009E73",
                 "jde": "#CC79A7", "coordinate": "#D55E00", "abom": "#E69F00",
                 "abom_perturb": "#56B4E9"}
FAMILY_LABELS = {"single_bump": "Single mode (4 variables)", "paired_bump": "Paired modes (7 variables)"}
FAMILY_COLORS = {"single_bump": "#0072B2", "paired_bump": "#D55E00"}
LAYOUT_LABELS = {"central": "Central obstacle", "early_offset": "Early, offset obstacle",
                 "two_obstacles": "Two obstacles"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 220,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.axisbelow": True})


def _close(actual, expected, name):
    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Summary arithmetic mismatch for {name}: {actual} != {expected}")


def load_campaign(output):
    """Check binding and recompute plotted means from the individual run rows."""
    manifest, selection = load_selection(output)
    for relative, expected in manifest["source_sha256"].items():
        if digest(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source snapshot mismatch: {relative}")
    # Statistical figures read frozen records. Only path reconstruction needs
    # current geometry code; a later runner-only audit repair does not invalidate
    # the earlier records or justify replacing their frozen hashes.
    root = Path(__file__).resolve().parents[1]
    for relative in ("python/trajectory_research/curves.py", "python/trajectory_research/parameterizations.py"):
        if digest(root / relative) != manifest["source_sha256"][relative]:
            raise ValueError(f"Current geometry differs from the frozen campaign: {relative}")
    summary = read(output / "summary.json")
    validation = read(output / "validation_summary.json")
    if summary["manifest_sha256"] != digest(output / "manifest.json") \
            or summary["validation_summary_sha256"] != digest(output / "validation_summary.json"):
        raise ValueError("Summary input binding mismatch")
    for filename, key in (("manifest.json", "manifest_sha256"),
                          ("selection.json", "selection_sha256"),
                          ("stress_cases.json", "stress_sha256")):
        if validation[key] != digest(output / filename):
            raise ValueError("Validation summary input binding mismatch")
    indexed = {(row["family"], row["layout"], row["method"], row["seed"]): row
               for row in validation["rows"]}
    if len(indexed) != len(validation["rows"]):
        raise ValueError("Duplicate validation rows")
    runs = []
    for chosen in selection["incumbents"]:
        identity = tuple(chosen[k] for k in ("family", "layout", "method", "seed"))
        row = indexed[identity]
        result_path = output / row["result_file"]
        if row["result_sha256"] != digest(result_path):
            raise ValueError("Validation result digest mismatch")
        result = read(result_path)
        if any(row[k] != result[k] for k in row if k not in ("result_file", "result_sha256")):
            raise ValueError("Validation row differs from the result")
        training = read(output / chosen["training_file"])
        if training["best_parameters"] != chosen["parameters"] or training["best_score"] != chosen["training_score"]:
            raise ValueError("Frozen selected incumbent mismatch")
        runs.append(dict(**row, elapsed_search_s=training["elapsed_s"],
            final_training_score=training["best_score"],
            training_feasible=bool(result["training_check"]["feasible"]),
            parameters=chosen["parameters"], history=training["history"],
            training_file=chosen["training_file"], training_sha256=chosen["training_sha256"]))
    expected_groups = {(f, l, m) for f in manifest["families"] for l in manifest["layouts"] for m in manifest["methods"]}
    actual_groups = [(g["family"], g["layout"], g["method"]) for g in summary["groups"]]
    if len(actual_groups) != len(expected_groups) or set(actual_groups) != expected_groups:
        raise ValueError("Summary group coverage differs from the campaign")
    for group in summary["groups"]:
        items = [r for r in runs if all(r[k] == group[k] for k in ("family", "layout", "method"))]
        if len(items) != group["runs"] or len(items) != len(manifest["seeds"]):
            raise ValueError("Group seed coverage mismatch")
        for field, original in (("mean_quality", "quality"), ("mean_duration_s", "duration_s"),
                                ("mean_search_s", "elapsed_search_s")):
            _close(np.mean([r[original] for r in items]), group[field], field)
        if sum(r["all_conditions_pass"] for r in items) != group["full_pass_runs"] \
                or sum(r["failures"] for r in items) != group["failed_conditions"]:
            raise ValueError("Group full-pass/failure arithmetic mismatch")
    return manifest, summary, runs


def _save(fig, destination, name, saved):
    for suffix in ("png", "pdf"):
        path = destination / f"{name}.{suffix}"
        kwargs = dict(bbox_inches="tight", pad_inches=.12)
        if suffix == "pdf":
            kwargs["metadata"] = {"Creator": "Matplotlib / forAirsim perturbation study", "CreationDate": None, "ModDate": None}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)


def _group_rows(runs, family, layout, method):
    return sorted([r for r in runs if (r["family"], r["layout"], r["method"]) == (family, layout, method)],
                  key=lambda r: r["seed"])


def quality_boxplots(manifest, runs, destination, saved):
    families, layouts, methods = manifest["families"], manifest["layouts"], manifest["methods"]
    fig, axes = plt.subplots(len(layouts), len(families), squeeze=False, sharey="row",
                             figsize=(6.1*len(families), 3.0*len(layouts)))
    for i, layout in enumerate(layouts):
        for j, family in enumerate(families):
            ax = axes[i, j]
            groups = [_group_rows(runs, family, layout, method) for method in methods]
            values = [[r["quality"] for r in group] for group in groups]
            boxes = ax.boxplot(values, tick_labels=[METHOD_LABELS[m] for m in methods],
                patch_artist=True, widths=.55, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="white", markersize=4),
                medianprops=dict(color="black", linewidth=1.1),
                flierprops=dict(marker="o", markersize=2.5, alpha=.35))
            for box, method in zip(boxes["boxes"], methods):
                box.set(facecolor=METHOD_COLORS[method], alpha=.4, linewidth=.8)
            for position, group in enumerate(groups, 1):
                failures = [r["quality"] for r in group if not r["all_conditions_pass"] or not r["training_feasible"]]
                if failures:
                    ax.scatter([position]*len(failures), failures, marker="x", color="#B2182B", zorder=6)
            ax.set_title(f"{LAYOUT_LABELS[layout]} | {FAMILY_LABELS[family]}")
            ax.set_ylabel("Reference quality (lower is better)")
            ax.tick_params(axis="x", labelrotation=28)
            ax.tick_params(axis="y", labelleft=True)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Initial-curve parameter search: seed distributions and means", y=.995, fontsize=14)
    fig.legend(handles=[Line2D([], [], marker="D", color="none", markerfacecolor="black", markersize=5, label="Mean"),
                        Line2D([], [], color="black", label="Median; box = interquartile range"),
                        Line2D([], [], marker="x", color="#B2182B", linestyle="none", label="Training/stress failure")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .025))
    fig.text(.5, .006, f"{len(manifest['seeds'])} optimizer seeds per cell; {manifest['budget']} candidate evaluations per run. Fixed development layouts.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .09, 1, .97])
    _save(fig, destination, "parameter_quality_boxplots", saved)


def mean_metrics(manifest, summary, destination, saved):
    layouts, families, methods = manifest["layouts"], manifest["families"], manifest["methods"]
    metrics = (("mean_quality", "Mean reference quality"), ("mean_duration_s", "Mean reference duration (s)"),
               ("mean_search_s", "Mean search wall time (s)"))
    fig, axes = plt.subplots(len(layouts), 3, squeeze=False, figsize=(15, 3.05*len(layouts)))
    x = np.arange(len(methods))
    for i, layout in enumerate(layouts):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            for family in families:
                indexed = {g["method"]: g for g in summary["groups"] if (g["family"], g["layout"]) == (family, layout)}
                ax.plot(x, [indexed[m][metric] for m in methods], marker="o", markersize=4,
                        linewidth=1.2, color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
            if metric == "mean_search_s":
                values = [g[metric] for g in summary["groups"] if g["layout"] == layout]
                if min(values) > 0 and max(values)/min(values) >= 10.:
                    ax.set_yscale("log")
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_title(LAYOUT_LABELS[layout])
            ax.set_ylabel(label)
            ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=28)
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(families), frameon=False, bbox_to_anchor=(.5, .02))
    fig.suptitle("Quality, reference duration, and search computation are separate outcomes", y=.995, fontsize=14)
    fig.text(.5, .005, "Search wall time is measured in this runtime; reference duration is a curve parameter, not an observed flight time.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .085, 1, .97])
    _save(fig, destination, "parameter_mean_metrics", saved)


def convergence(manifest, runs, destination, saved):
    families, layouts, methods = manifest["families"], manifest["layouts"], manifest["methods"]
    fig, axes = plt.subplots(len(layouts), len(families), squeeze=False, sharey="row",
                             figsize=(6.0*len(families), 2.9*len(layouts)))
    for i, layout in enumerate(layouts):
        for j, family in enumerate(families):
            ax = axes[i, j]
            for method in methods:
                group = _group_rows(runs, family, layout, method)
                fe = np.array([r["fe"] for r in group[0]["history"]])
                if not all([r["fe"] for r in item["history"]] == fe.tolist() for item in group):
                    raise ValueError("History FE grids differ")
                values = np.array([[row["best"] for row in item["history"]] for item in group])
                low, median, high = np.quantile(values, [.25, .5, .75], axis=0)
                color = METHOD_COLORS[method]
                ax.fill_between(fe, low, high, color=color, alpha=.09, linewidth=0, step="post")
                ax.step(fe, median, color=color, linewidth=1.4, label=METHOD_LABELS[method], where="post")
            ax.set_title(f"{LAYOUT_LABELS[layout]} | {FAMILY_LABELS[family]}")
            ax.set_xlabel("Cumulative candidate evaluations (FE)")
            ax.set_ylabel("Best training score (lower)")
            ax.tick_params(axis="y", labelleft=True)
            ax.set_xlim(manifest["population"], manifest["budget"])
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(methods)), frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Training progress under an equal candidate-evaluation budget", y=.995, fontsize=14)
    fig.text(.5, .006, "Lines: median across optimizer seeds. Bands: interquartile range. Shared initialization within each family and layout.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .09, 1, .97])
    _save(fig, destination, "parameter_convergence", saved)


def _projected_envelope(centre, factor, radius, dims):
    """Boundary of projected (ellipsoid + closed ball), using support points."""
    theta = np.linspace(0, 2*np.pi, 361)
    normal = np.stack((np.cos(theta), np.sin(theta)))
    projected = factor[list(dims), :]
    dual = projected.T @ normal
    length = np.linalg.norm(dual, axis=0)
    ellipsoid_point = np.divide(projected @ dual, length, out=np.zeros_like(normal), where=length > 0)
    return (np.asarray(centre)[list(dims), None] + ellipsoid_point + radius*normal).T


def _envelope_surface(centre, factor, radius):
    azimuth, polar = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, np.pi, 18))
    normal = np.array((np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth), np.cos(polar)))
    flat = normal.reshape(3, -1)
    dual = factor.T @ flat
    length = np.linalg.norm(dual, axis=0)
    ellipsoid = np.divide(factor @ dual, length, out=np.zeros_like(flat), where=length > 0)
    return (np.asarray(centre)[:, None]+ellipsoid+radius*flat).reshape(normal.shape)


def representative_paths(manifest, runs, destination, saved, method):
    chosen_rows = []
    factor = np.array(manifest["task"]["geometry_factor_m"])
    scenes = {x["name"]: x for x in manifest["task"]["layouts"]}
    u = np.linspace(0, 1, 401)
    for layout in manifest["layouts"]:
        fig = plt.figure(figsize=(14.4, 4.8))
        xy = fig.add_subplot(1, 3, 1)
        xz = fig.add_subplot(1, 3, 2)
        xyz = fig.add_subplot(1, 3, 3, projection="3d")
        curves = []
        first_family = manifest["families"][0]
        initial_parameters = manifest["variables"][first_family]["anchor"]
        initial = curve_from_parameters(initial_parameters, family=first_family)
        curves.append(("Initial warm start", initial.evaluate(u), "#555555", "--"))
        for family in manifest["families"]:
            family_anchor = curve_from_parameters(manifest["variables"][family]["anchor"], family=family)
            if not np.allclose(family_anchor.evaluate(u), curves[0][1], rtol=0, atol=1e-10):
                raise ValueError("One initial path would conceal distinct family warm starts")
            group = sorted(_group_rows(runs, family, layout, method),
                           key=lambda row: (row["final_training_score"], row["seed"]))
            row = group[(len(group)-1)//2]
            curve = curve_from_parameters(row["parameters"], family=family)
            label = f"{FAMILY_LABELS[family].split(' (')[0]}: seed {row['seed']}, T={row['duration_s']:.2f} s"
            curves.append((label, curve.evaluate(u), FAMILY_COLORS[family], "-"))
            chosen_rows.append({k: row[k] for k in ("family", "layout", "method", "seed", "parameters",
                "final_training_score", "duration_s", "training_file", "training_sha256")})
        for obstacle in scenes[layout]["obstacles"]:
            centre, radius = obstacle["centre"], obstacle["radius"]
            total_radius = radius + manifest["task"]["radius_increment_m"] + manifest["task"]["clearance_m"]
            for ax, dims in ((xy, (0, 1)), (xz, (0, 2))):
                polygon = _projected_envelope(centre, factor, total_radius, dims)
                ax.add_patch(Polygon(polygon, facecolor="#C9C9C9", edgecolor="#777777", linewidth=.8, alpha=.45))
                ax.add_patch(Circle(np.asarray(centre)[list(dims)], radius, facecolor="#999999", edgecolor="#555555", alpha=.35))
            surface = _envelope_surface(centre, factor, total_radius)
            xyz.plot_surface(*surface, color="#999999", alpha=.13, linewidth=0, shade=False)
        for label, values, color, linestyle in curves:
            for ax, dims in ((xy, (0, 1)), (xz, (0, 2))):
                ax.plot(values[:, dims[0]], values[:, dims[1]], color=color, linestyle=linestyle,
                        linewidth=1.8, label=label)
            xyz.plot(values[:, 0], values[:, 1], values[:, 2], color=color, linestyle=linestyle, linewidth=1.8)
        workspace = np.array(manifest["task"]["workspace"])
        for ax, dims in ((xy, (0, 1)), (xz, (0, 2))):
            ax.set(xlabel="x (m)", ylabel="y (m)" if dims[1] == 1 else "z (m)",
                   xlim=workspace[:, dims[0]], ylim=workspace[:, dims[1]])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.18)
            ax.scatter([curves[0][1][0, 0], curves[0][1][-1, 0]],
                       [curves[0][1][0, dims[1]], curves[0][1][-1, dims[1]]], s=14, color="black", zorder=7)
        xy.set_title("XY projection")
        xz.set_title("XZ projection")
        xyz.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="3-D point reference",
                xlim=workspace[:, 0], ylim=workspace[:, 1], zlim=workspace[:, 2])
        xyz.set_box_aspect(workspace[1]-workspace[0])
        xyz.view_init(elev=22, azim=-56)
        xyz.set_xticks([0., 2., 4., 6., 8.])
        xyz.set_yticks([-1.5, 0., 1.5])
        xyz.set_zticks([3.5, 4.5, 5.5, 6.5])
        xyz.tick_params(labelsize=8)
        handles, labels = xy.get_legend_handles_labels()
        handles.append(Patch(facecolor="#C9C9C9", edgecolor="#777777", alpha=.5))
        labels.append("Modeled obstacle envelope + clearance")
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .08), fontsize=9)
        fig.suptitle(f"{LAYOUT_LABELS[layout]} | {METHOD_LABELS[method]} | lower-median training seed per family", y=.985, fontsize=13)
        fig.text(.5, .025, "Coupled Hermite/Bézier decoders describe the same curves. Endpoint p/v/a are fixed. Projected overlap does not imply 3-D collision.",
                 ha="center", fontsize=8.8, color="#555555")
        fig.tight_layout(rect=[0, .20, 1, .97])
        _save(fig, destination, f"parameter_paths_{layout}", saved)
    return chosen_rows


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Frozen campaign directory")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--path-method", choices=list(METHOD_LABELS), default=None,
                        help="Fixed method for path examples; default coordinate, then Gaussian, then first declared")
    args = parser.parse_args()
    output, destination = args.output.resolve(), args.destination.resolve()
    manifest, summary, runs = load_campaign(output)
    method = args.path_method or ("coordinate" if "coordinate" in manifest["methods"] else
                                 "gaussian" if "gaussian" in manifest["methods"] else manifest["methods"][0])
    if method not in manifest["methods"]:
        parser.error("path method is not present in the campaign")
    destination.mkdir(parents=True, exist_ok=True)
    _style()
    saved = []
    quality_boxplots(manifest, runs, destination, saved)
    mean_metrics(manifest, summary, destination, saved)
    convergence(manifest, runs, destination, saved)
    selected = representative_paths(manifest, runs, destination, saved, method)
    paths = [("parameter_group_metrics.csv", summary["groups"],
              ("family", "layout", "method", "runs", "mean_quality", "mean_duration_s", "mean_search_s",
               "full_pass_runs", "failed_conditions", "mean_jerk_energy_m2_s5", "mean_final_training_score")),
             ("parameter_runs.csv", runs,
              ("family", "layout", "method", "seed", "quality", "duration_s", "elapsed_search_s",
               "training_feasible", "all_conditions_pass", "conditions", "failures", "jerk_energy_m2_s5",
               "deformation_rms_m", "final_training_score", "parameters", "training_file", "training_sha256")),
             ("parameter_selected_paths.csv", selected,
              ("family", "layout", "method", "seed", "final_training_score", "duration_s", "parameters",
               "training_file", "training_sha256"))]
    for name, rows, fields in paths:
        path = destination / name
        _csv(path, rows, fields)
        saved.append(path)
    input_paths = [output / name for name in ("manifest.json", "selection.json", "summary.json", "validation_summary.json")]
    provenance = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), campaign=str(output),
        input_sha256={path.name: digest(path) for path in input_paths},
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        path_selection=dict(method=method, rule="Sort each family/layout's training scores, then seeds; take index (n-1)//2",
                            validation_used=False, selected=selected),
        figures_and_tables_sha256={path.name: digest(path) for path in saved},
        scope=summary["scope"],
        interpretation="Optimizer seeds are the statistical units. Envelopes are assumed, not calibrated. No flight performance is measured.")
    (destination / "parameter_plot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(campaign=str(output), files=len(saved)+1, destination=str(destination), path_method=method)))


if __name__ == "__main__":
    main()

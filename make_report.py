"""Rebuild figures, CSV tables and clustered uncertainty from cached predictions."""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from compass import metrics
from evaluate_predictions import evaluate
from run_benchmark import json_write


def load_case(row, method):
    cache = Path("cache") / row["case"]
    with np.load(cache / "samples.npz") as handle:
        samples = dict(handle)
    with np.load(cache / f"{method}.npz") as handle:
        prediction = dict(handle)
    return samples, prediction


def risk_curve(error, confidence, weight):
    order = np.argsort(-confidence, kind="stable")
    w = weight[order]
    coverage = np.cumsum(w) / w.sum()
    risk = np.cumsum(w * (error[order] > 30)) / np.cumsum(w)
    return coverage, risk


def bootstrap(cases, method, threshold):
    clusters = []
    for row in cases:
        samples, selected = load_case(row, method)
        _, baseline = load_case(row, "sigma1_rho1")
        ids = samples["fiber_ids"]
        table = []
        for ident in np.unique(ids):
            take = ids == ident
            a = selected["confidence"][take] >= threshold
            e = selected["errors"][take]
            b = baseline["errors"][take]
            table.append([b.mean() - e.mean(), a.mean(), (a * e).mean(), (a * (e > 30)).mean()])
        clusters.append(np.array(table))
    rng = np.random.default_rng(20260924)
    values = []
    for _ in range(2000):
        cube_rows = []
        for case_index in rng.integers(0, len(clusters), size=len(clusters)):
            table = clusters[case_index]
            cube_rows.append(table[rng.integers(0, len(table), size=len(table))].mean(axis=0))
        delta, coverage, error_mass, bad_mass = np.mean(cube_rows, axis=0)
        values.append([delta, coverage, error_mass / coverage, bad_mass / coverage])
    return {name: np.percentile(np.array(values)[:, i], [2.5, 97.5]).tolist()
            for i, name in enumerate(["mean_error_reduction_degrees", "retained_coverage", "retained_mean_degrees", "retained_over_30"])}


def main():
    developed = json.loads(Path("results/develop.json").read_text())
    evaluated = json.loads(Path("results/evaluate.json").read_text())
    selection = evaluated["selection"]
    method, threshold = selection["method"], selection["confidence_threshold"]
    if threshold is None:
        raise RuntimeError("The report expects a development-calibrated threshold")
    plotdir = Path("results/figures")
    plotdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cases = evaluated["cases"]
    x = np.arange(len(cases))
    ax[0, 0].bar(x - .18, [c["methods"]["sigma1_rho1"]["mean_degrees"] for c in cases], .36,
                  label="Reference: sigma 1, rho 1", color="#87949f")
    ax[0, 0].bar(x + .18, [c["methods"][method]["mean_degrees"] for c in cases], .36,
                  label="Selected on Scroll 1: sigma 1.5, rho 3", color="#007e87")
    ax[0, 0].set(xticks=x, xticklabels=[f"S5-{i+1}" for i in x], ylabel="Mean angular error (degrees)",
                  title="A. Improvement on all six evaluation cubes", ylim=(0, 48))
    ax[0, 0].legend(fontsize=8)
    combined = {}
    for name, color, label in [("sigma1_rho1", "#87949f", "Reference"), (method, "#007e87", "Selected on Scroll 1")]:
        errors, confidence, weights, tangents = [], [], [], []
        for case in cases:
            s, p = load_case(case, name)
            errors.append(p["errors"])
            confidence.append(p["confidence"])
            weights.append(s["weights"] / len(cases))
            tangents.append(s["tangents_zyx"])
        e, c, w, t = map(np.concatenate, (errors, confidence, weights, tangents))
        combined[name] = e, c, w, t
        coverage, risk = risk_curve(e, c, w)
        take = np.linspace(0, len(coverage) - 1, min(600, len(coverage))).astype(int)
        ax[0, 1].plot(coverage[take] * 100, risk[take] * 100, label=label, color=color)
    frozen = evaluated["frozen_threshold_result"]
    ax[0, 1].scatter(frozen["coverage"] * 100, frozen["over_30_degrees"] * 100,
                     marker="*", s=120, color="#b75b32", label="Frozen confidence threshold 0.75", zorder=5)
    ax[0, 1].set(xlabel="Retained evaluation weight (%)", ylabel="Errors above 30 degrees (%)",
                 title="B. Confidence-ranked risk versus coverage", xlim=(0, 100), ylim=(0, 60))
    ax[0, 1].legend(fontsize=8, loc="upper left")
    thresholds = list(evaluated["confidence_risk"])
    xx = np.arange(len(thresholds))
    for offset, summary, color, label in [(-.18, selection["confidence_risk"], "#c7a568", "Scroll 1: development"),
                                          (.18, evaluated["confidence_risk"], "#007e87", "Scroll 5: evaluation")]:
        ax[1, 0].bar(xx + offset, [summary[t]["coverage"] * 100 for t in thresholds], .36, color=color, label=label)
    ax[1, 0].set(xticks=xx, xticklabels=thresholds, xlabel="Minimum eigenvalue-gap confidence",
                 ylabel="Retained weight (%)", title="C. A fixed threshold retains much less on Scroll 5")
    ax[1, 0].legend(fontsize=8)
    e, c, w, t = combined[method]
    strata = [("Near Z", np.abs(t[:, 0]) >= np.cos(np.pi/6)),
              ("Near XY", np.abs(t[:, 0]) <= .5),
              ("Oblique", (np.abs(t[:, 0]) > .5) & (np.abs(t[:, 0]) < np.cos(np.pi/6)))]
    stratified = {}
    for name, mask in strata:
        stratified[name] = {"all": metrics(e[mask], w[mask]),
                            "retained": metrics(e[mask], w[mask], c[mask] >= threshold),
                            "population_weight": float(w[mask].sum())}
    ax[1, 1].bar(np.arange(3), [stratified[name]["retained"]["coverage"] * 100 for name, _ in strata], color="#b75b32")
    ax[1, 1].set(xticks=np.arange(3), xticklabels=[name for name, _ in strata],
                 ylabel="Within-group retained weight (%)", title="D. Retention differs by reference orientation")
    fig.suptitle("Fiber Compass | Manual fiber directions across two scrolls", fontsize=17, fontweight="bold", y=.99)
    fig.text(.5, .015, "Equal cubes and equal eligible fibers. Unoriented angle: v and -v are equivalent.\n"
                      "Local direction agreement only; no measured improvement in connectivity, unwrapping, or text recovery.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(plotdir / "validation.png", dpi=160)
    plt.close(fig)
    # Inspect deterministic examples: a median-error retained point, then the
    # maximum-error retained point in the first evaluation cube. No deletion.
    row = cases[0]
    example_case = row["case"]
    samples, prediction = load_case(row, method)
    eligible = np.flatnonzero(prediction["confidence"] >= threshold)
    ordered = eligible[np.argsort(prediction["errors"][eligible], kind="stable")]
    chosen = [ordered[len(ordered)//2], ordered[-1]]
    image = tifffile.imread(Path("data") / row["image"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, index in enumerate(chosen):
        p = samples["positions_zyx"][index]
        ref = samples["tangents_zyx"][index]
        pred = prediction["axes_zyx"][index]
        for column, (fixed, horizontal, vertical, name) in enumerate([(0, 2, 1, "XY"), (1, 2, 0, "XZ"), (2, 1, 0, "YZ")]):
            sl = np.take(image, int(round(p[fixed])), axis=fixed)
            axes[r, column].imshow(sl, cmap="gray", vmin=0, vmax=255, origin="lower")
            for vector, color, label in [(ref, "#38d8ff", "Manual tangent"), (pred, "#ff9b45", "Predicted axis")]:
                d = vector * 15
                axes[r, column].plot([p[horizontal]-d[horizontal], p[horizontal]+d[horizontal]],
                                     [p[vertical]-d[vertical], p[vertical]+d[vertical]], color=color, lw=2, label=label)
            axes[r, column].set(xlim=(p[horizontal]-25, p[horizontal]+25), ylim=(p[vertical]-25, p[vertical]+25),
                                 title=f"{name}: {'median retained' if r == 0 else 'worst retained'}")
            axes[r, column].set_aspect("equal")
        axes[r, 0].set_ylabel(f"3D error {prediction['errors'][index]:.1f} degrees\nconfidence {prediction['confidence'][index]:.3f}")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(f"A useful cue and a confident failure | {row['case']}", fontsize=15)
    fig.text(.5, .012, "CT and annotations: Vesuvius Challenge Dataset001, 7.91 micrometers/voxel (CC BY-NC 4.0).\n"
                      "Each panel is an orthogonal slice through the query. Projected line lengths preserve 3D component magnitudes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .96))
    fig.savefig(plotdir / "real_ct_examples.png", dpi=160)
    plt.close(fig)
    audit = Counter()
    for row in developed["cases"] + cases:
        audit.update(row["audit"])
    intervals = bootstrap(cases, method, threshold)
    same_subset = metrics(combined["sigma1_rho1"][0], w, c >= threshold)
    json_write("results/analysis.json", {"annotation_audit": dict(audit), "orientation_strata": stratified,
                "reference_on_selected_retained_subset": same_subset,
                "cluster_bootstrap_95_percent_intervals": intervals,
                "bootstrap_scope": "Resample cubes, then fibers within each selected cube; 2000 repeats, seed 20260924. Conditional on these six cubes from one evaluation scroll, not six independent scrolls.",
                "ct_examples": [{"case": example_case, "sample_index": int(i)} for i in chosen]})
    with Path("results/per_cube.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "case", "eligible_fibers", "samples", "reference_mean_deg", "selected_mean_deg", "retained_coverage", "retained_mean_deg", "retained_over_30"])
        for stage in [developed, evaluated]:
            for row in stage["cases"]:
                m = row["methods"][method]
                retained = m["thresholds"][str(threshold)]
                writer.writerow([stage["stage"], row["case"], row["audit"]["eligible_paths"], row["samples"],
                                 row["methods"]["sigma1_rho1"]["mean_degrees"], m["mean_degrees"],
                                 retained["coverage"], retained["mean_degrees"], retained["over_30_degrees"]])
                s, p = load_case(row, method)
                p["positions_zyx"] = s["positions_zyx"]
                report = evaluate(s, p)
                json_write(Path("results/triage") / f"{row['case']}.json", report)
    print(json.dumps({"audit": dict(audit), "intervals": intervals, "strata": stratified,
                      "reference_same_subset": same_subset}, indent=2))


if __name__ == "__main__":
    main()

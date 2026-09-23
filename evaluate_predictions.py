"""Score external direction predictions at the benchmark's reference locations.

NPZ contract: positions_zyx (N,3), axes_zyx (N,3), confidence (N,).
Positions must exactly match cache/CASE/samples.npz; no implicit registration,
axis permutation, reordering, or missing-point deletion is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from compass import angular_error, metrics
from run_benchmark import json_write


def evaluate(reference, prediction, thresholds=(0, .25, .5, .75, .9)):
    expected = reference["positions_zyx"]
    positions = np.asarray(prediction["positions_zyx"], dtype=float)
    axes = np.asarray(prediction["axes_zyx"], dtype=float)
    confidence = np.asarray(prediction["confidence"], dtype=float)
    if positions.shape != expected.shape or not np.allclose(positions, expected, atol=1e-5, rtol=0):
        raise ValueError("Prediction locations/order differ from the reference; explicit registration is required")
    if axes.shape != expected.shape or confidence.shape != (len(expected),):
        raise ValueError("Expected axes_zyx (N,3) and confidence (N,)")
    if not np.isfinite(positions).all() or not np.isfinite(confidence).all():
        raise ValueError("Locations/confidences must be finite")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must lie in [0,1]")
    valid = np.isfinite(axes).all(axis=1) & (np.linalg.norm(axes, axis=1) > 1e-12)
    # Invalid/missing predictions count as maximal error in all-sample scores.
    # They can be abstentions in thresholded scores, but never vanish silently.
    errors = np.full(len(expected), 90.)
    errors[valid] = angular_error(axes[valid], reference["tangents_zyx"][valid])
    weights = reference["weights"]
    report = {"samples": len(expected), "invalid_predictions": int((~valid).sum()),
              "all_samples": metrics(errors, weights),
              "thresholds": {str(t): metrics(errors, weights, valid & (confidence >= t)) for t in thresholds}}
    report["worst_fibers"] = []
    for ident in np.unique(reference["fiber_ids"]):
        select = reference["fiber_ids"] == ident
        record = metrics(errors[select], weights[select])
        record["fiber_index"] = int(ident)
        record["invalid_predictions"] = int((select & ~valid).sum())
        record["high_confidence_over_30"] = int((select & valid & (confidence >= .75) & (errors > 30)).sum())
        report["worst_fibers"].append(record)
    report["worst_fibers"].sort(key=lambda r: (-r["mean_degrees"], r["fiber_index"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as reference, np.load(args.prediction, allow_pickle=False) as prediction:
        report = evaluate(reference, prediction)
    json_write(args.output, report)
    print(json.dumps(report["all_samples"], indent=2))


if __name__ == "__main__":
    main()

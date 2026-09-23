"""Run development first, freeze selection, then evaluate an untouched scroll."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
import tifffile

from compass import read_paths, sample_paths, sparse_tensor, decompose, angular_error, metrics
from download_data import cases


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def key(sigma, rho):
    return f"sigma{sigma:g}_rho{rho:g}"


def json_write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run_case(row, candidates, protocol):
    start = time.perf_counter()
    image = tifffile.imread(Path("data") / row["image"])
    if image.shape != (row["size"],) * 3 or image.dtype != np.uint8:
        raise ValueError(f"Unexpected CT shape or dtype: {row['case']}")
    paths, audit, voxel_scale = read_paths(Path("data") / row["nml"])
    samples = sample_paths(paths, row["origin_xyz"], image.shape,
                           spacing=protocol["sample_spacing_voxels"],
                           half_window=protocol["tangent_half_window_voxels"],
                           border=protocol["common_border_voxels"], audit=audit)
    grouped = defaultdict(list)
    for sigma, rho in candidates:
        grouped[sigma].append(rho)
    cache = Path("cache") / row["case"]
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / "samples.npz", positions_zyx=samples.positions_zyx,
                        tangents_zyx=samples.tangents_zyx, fiber_ids=samples.fiber_ids, weights=samples.weights)
    methods = {}
    arrays = {}
    for sigma, rhos in grouped.items():
        for rho, tensor in sparse_tensor(image, samples.positions_zyx, sigma, rhos,
                                         truncate=protocol["gaussian_truncate"]):
            axis, confidence, values = decompose(tensor)
            errors = angular_error(axis, samples.tangents_zyx)
            name = key(sigma, rho)
            methods[name] = metrics(errors, samples.weights)
            methods[name]["thresholds"] = {
                str(t): metrics(errors, samples.weights, confidence >= t)
                for t in protocol["confidence_thresholds"]
            }
            arrays[name] = (errors, confidence, samples.weights)
            np.savez_compressed(cache / f"{name}.npz", errors=errors, confidence=confidence,
                                axes_zyx=axis, eigenvalues=values)
            print(row["case"], name, "mean_deg", round(methods[name]["mean_degrees"], 3), flush=True)
    # Three deterministic CT-independent baselines, chosen on development only.
    for i, axis_name in enumerate("zyx"):
        axes = np.zeros_like(samples.tangents_zyx)
        axes[:, i] = 1
        errors = angular_error(axes, samples.tangents_zyx)
        methods[f"fixed_{axis_name}"] = metrics(errors, samples.weights)
        arrays[f"fixed_{axis_name}"] = (errors, np.ones(len(errors)), samples.weights)
    record = {**row, "image_sha256": sha(Path("data") / row["image"]),
              "nml_sha256": sha(Path("data") / row["nml"]),
              "voxel_size": voxel_scale.tolist(), "audit": samples.audit,
              "samples": len(samples.positions_zyx), "methods": methods,
              "seconds": time.perf_counter() - start}
    return record, arrays


def combine(arrays, name, threshold=0):
    errors = np.concatenate([x[name][0] for x in arrays])
    confidence = np.concatenate([x[name][1] for x in arrays])
    weights = np.concatenate([x[name][2] / len(arrays) for x in arrays])
    return metrics(errors, weights, confidence >= threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    args = parser.parse_args()
    protocol = json.loads(Path("protocol.json").read_text())
    if protocol["bad_angle_degrees"] != 30:
        raise ValueError("This benchmark reports the fixed 30-degree failure criterion")
    protocol_hash = sha("protocol.json")
    if args.stage == "develop":
        candidates = protocol["parameter_candidates"]
        scroll = protocol["development_scroll"]
        if Path("results/evaluate.json").exists():
            raise RuntimeError("Evaluation already exists: do not silently retune after observing it")
    else:
        selection = json.loads(Path("results/selection.json").read_text())
        if selection["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Protocol changed after development")
        if selection["compass_sha256"] != sha("compass.py"):
            raise RuntimeError("Method changed after development")
        candidates = [protocol["reference_parameters"]]
        if selection["parameters"] not in candidates:
            candidates.append(selection["parameters"])
        scroll = protocol["evaluation_scroll"]
    records, arrays = [], []
    for row in cases():
        if row["scroll"] == scroll:
            record, array = run_case(row, candidates, protocol)
            records.append(record)
            arrays.append(array)
            json_write(f"results/{args.stage}_partial.json", records)
    aggregate = {name: combine(arrays, name) for name in arrays[0]}
    summary = {"stage": args.stage, "created_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": protocol_hash, "compass_sha256": sha("compass.py"),
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "tifffile": tifffile.__version__},
               "weighting": "Equal cubes; within a cube, equal total weight for each eligible fiber",
               "cases": records, "aggregate": aggregate}
    if args.stage == "develop":
        best = min(candidates, key=lambda p: aggregate[key(*p)]["mean_degrees"])
        best_name = key(*best)
        risks = {str(t): combine(arrays, best_name, t) for t in protocol["confidence_thresholds"]}
        feasible = [t for t in protocol["confidence_thresholds"]
                    if risks[str(t)]["over_30_degrees"] is not None
                    and risks[str(t)]["over_30_degrees"] <= protocol["calibration_target_bad_fraction"]]
        threshold = min(feasible) if feasible else None
        selection = {"parameters": best, "method": best_name, "confidence_threshold": threshold,
                     "calibration_met_target": threshold is not None, "confidence_risk": risks,
                     "fixed_axis": min((f"fixed_{a}" for a in "zyx"), key=lambda n: aggregate[n]["mean_degrees"]),
                     "protocol_sha256": protocol_hash, "compass_sha256": sha("compass.py"),
                     "frozen_utc": datetime.now(timezone.utc).isoformat()}
        json_write("results/selection.json", selection)
        print("FROZEN", json.dumps(selection), flush=True)
    else:
        summary["selection"] = selection
        summary["confidence_risk"] = {str(t): combine(arrays, selection["method"], t)
                                        for t in protocol["confidence_thresholds"]}
        threshold = selection["confidence_threshold"]
        summary["frozen_threshold_result"] = (None if threshold is None else
                                                combine(arrays, selection["method"], threshold))
    json_write(f"results/{args.stage}.json", summary)
    print("AGGREGATE", json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()

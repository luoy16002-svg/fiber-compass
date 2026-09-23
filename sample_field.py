"""Sample a Fiber Compass Zarr field at NPZ query points, respecting axis sign."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def sample(group, positions_zyx, reference_origin_xyz):
    metadata = dict(group.attrs)
    if metadata.get("format") != "fiber-compass-1" or not metadata.get("complete"):
        raise ValueError("Not a complete Fiber Compass field")
    if metadata.get("axes") != "ZYX" or metadata.get("projector_components") != ["zz", "zy", "zx", "yy", "yx", "xx"]:
        raise ValueError("Unsupported field axis/component convention")
    origin_delta = np.asarray(reference_origin_xyz)[::-1] - np.asarray(metadata["source_origin_xyz"])[::-1]
    stride = float(metadata["grid_stride_source_voxels"])
    points = (np.asarray(positions_zyx) + origin_delta) / stride
    projector = group["axis_projector"]
    confidence = group["confidence"]
    shape = np.asarray(confidence.shape)
    valid = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) & (points <= shape - 1).all(axis=1)
    axes = np.full((len(points), 3), np.nan, dtype=np.float64)
    score = np.zeros(len(points), dtype=np.float64)
    ids = np.flatnonzero(valid)
    cells = np.floor(points[valid]).astype(np.int64) // np.array(confidence.chunks)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    tri = np.triu_indices(3)
    for i, cell in enumerate(unique):
        selected = ids[inverse == i]
        lower = cell * confidence.chunks
        upper = np.minimum(lower + np.array(confidence.chunks) + 1, shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        query = (points[selected] - lower).T
        packed = np.asarray(projector[sl])
        values = np.stack([map_coordinates(packed[..., c], query, order=1, prefilter=False, mode="nearest") for c in range(6)], axis=1)
        matrices = np.zeros((len(selected), 3, 3))
        matrices[:, tri[0], tri[1]] = values
        matrices[:, tri[1], tri[0]] = values
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
        axes[selected] = eigenvectors[:, :, -1]
        q = map_coordinates(np.asarray(confidence[sl]), query, order=1, prefilter=False, mode="nearest")
        agreement = np.clip((eigenvalues[:, 2] - eigenvalues[:, 1]) / (eigenvalues[:, 2] + 1e-12), 0, 1)
        score[selected] = q * agreement
    return axes, score


def main():
    import zarr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-origin-xyz", type=float, nargs=3, required=True,
                        help="Global XYZ origin of local query coordinates, in the field's source voxels")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    with np.load(args.queries, allow_pickle=False) as handle:
        positions = handle["positions_zyx"]
    group = zarr.open_group(str(args.field), mode="r")
    axes, confidence = sample(group, positions, args.reference_origin_xyz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, positions_zyx=positions, axes_zyx=axes, confidence=confidence,
                        metadata_json=np.asarray(json.dumps({"source_field": str(args.field),
                            "confidence": "Interpolated gap score times projector eigengap agreement; not a correctness probability",
                            "reference_origin_xyz": args.reference_origin_xyz})))
    print(f"Wrote {len(positions)} predictions; {int(np.isfinite(axes).all(axis=1).sum())} in bounds")


if __name__ == "__main__":
    main()

"""Estimate a chunked, sign-invariant fiber-axis field from a uint8 CT TIFF.

Produces ordinary Zarr v2 arrays, with explicit voxel-coordinate metadata.
This is a local orientation cue, not a fiber detector or a complete tracer.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import tifffile

from compass import sparse_tensor, decompose


def tensor_blocks(image, sigma, rho, *, stride=4, block=96, truncate=4):
    if stride <= 0 or block <= 0 or block % stride:
        raise ValueError("block must be a positive multiple of stride")
    if sigma <= 0 or rho <= 0:
        raise ValueError("sigma and rho must be positive")
    halo = int(truncate * sigma + .5) + int(truncate * rho + .5) + 1
    for start in itertools.product(*(range(0, size, block) for size in image.shape)):
        start = np.array(start)
        end = np.minimum(start + block, image.shape)
        lower = np.maximum(start - halo, 0)
        upper = np.minimum(end + halo, image.shape)
        tile = image[tuple(slice(a, b) for a, b in zip(lower, upper))]
        grid = np.stack(np.meshgrid(*(np.arange(a, b, stride) for a, b in zip(start, end)), indexing="ij"), axis=-1)
        shape = grid.shape[:-1]
        positions = grid.reshape(-1, 3) - lower
        _, tensors = next(sparse_tensor(tile, positions, sigma, [rho], truncate=truncate))
        axes, confidence, values = decompose(tensors)
        # v v^T is unchanged by v -> -v. Interpolate these projectors, never
        # arbitrary-signed vectors, then take the principal eigenvector.
        projector = axes[:, :, None] * axes[:, None, :]
        tri = np.triu_indices(3)
        packed = projector[:, tri[0], tri[1]].reshape(*shape, 6).astype(np.float32)
        selection = tuple(slice(int(a // stride), int((b - 1) // stride + 1)) for a, b in zip(start, end))
        yield selection, packed, confidence.reshape(shape).astype(np.float32), values.reshape(*shape, 3).astype(np.float32)


def main():
    import zarr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=3)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--block", type=int, default=96)
    parser.add_argument("--origin-xyz", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--voxel-micrometers", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.voxel_micrometers <= 0:
        raise ValueError("Voxel size must be positive")
    if args.stride <= 0 or args.block <= 0 or args.block % args.stride:
        raise ValueError("block must be a positive multiple of stride")
    image = tifffile.imread(args.image)
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a 3D uint8 TIFF in ZYX order; convert explicitly first")
    with args.image.open("rb") as handle:
        image_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    shape = tuple((n + args.stride - 1) // args.stride for n in image.shape)
    chunk = tuple(min(n, args.block // args.stride) for n in shape)
    out = zarr.open_group(str(args.output), mode="w", zarr_format=2)
    projector = out.create_array("axis_projector", shape=(*shape, 6), chunks=(*chunk, 6), dtype="float32")
    confidence = out.create_array("confidence", shape=shape, chunks=chunk, dtype="float32")
    eigenvalues = out.create_array("eigenvalues", shape=(*shape, 3), chunks=(*chunk, 3), dtype="float32")
    out.attrs.update({"format": "fiber-compass-1", "complete": False,
                      "axes": "ZYX", "projector_components": ["zz", "zy", "zx", "yy", "yx", "xx"],
                      "source_origin_xyz": args.origin_xyz, "source_shape_zyx": list(image.shape),
                      "grid_stride_source_voxels": args.stride, "source_voxel_micrometers": args.voxel_micrometers,
                      "sigma_voxels": args.sigma, "rho_voxels": args.rho, "source_sha256": image_hash,
                      "confidence_definition": "(lambda_middle-lambda_min)/(lambda_middle+lambda_min+1e-12)",
                      "confidence_is_probability": False,
                      "scope": "Orientation only. Not fiber presence, connectivity, or correctness probability.",
                      "validity": "Calibrated only at annotated fiber locations on Dataset001; low-texture/background voxels are not validated."})
    projector.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x", "component"]
    confidence.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    eigenvalues.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x", "eigenvalue"]
    start = time.perf_counter()
    for number, (selection, p, c, v) in enumerate(tensor_blocks(image, args.sigma, args.rho,
                                                             stride=args.stride, block=args.block), 1):
        projector[selection] = p
        confidence[selection] = c
        eigenvalues[selection] = v
        print(f"wrote block {number}", flush=True)
    out.attrs.update({"complete": True, "seconds": time.perf_counter() - start})
    print(json.dumps(dict(out.attrs), indent=2))


if __name__ == "__main__":
    main()

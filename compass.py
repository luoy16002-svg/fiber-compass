"""Sparse, sign-invariant direction evaluation on manually traced CT fibers.

Arrays use ZYX; NML coordinates and origins use XYZ. Eigenvectors represent
unoriented axes: v and -v are identical predictions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class Samples:
    positions_zyx: np.ndarray
    tangents_zyx: np.ndarray
    fiber_ids: np.ndarray
    weights: np.ndarray
    audit: dict


def read_paths(path: str | Path):
    """Read connected open paths using graph edges, never XML node order.

    Branched/cyclic components are counted and excluded rather than silently
    inventing a route. A disconnected thing can contain multiple valid paths.
    """
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    scale = root.find("./parameters/scale")
    if scale is None:
        raise ValueError("NML is missing voxel scale")
    spacing = np.array([float(scale.attrib[a]) for a in "xyz"])
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid NML voxel scale")
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("This voxel-space benchmark requires isotropic CT")
    audit = Counter(things=0, nodes=0, components=0, branch_components=0,
                    cyclic_components=0, singleton_components=0, empty_things=0,
                    valid_paths=0, duplicate_edges=0, zero_length_edges=0)
    paths = []
    for thing in root.findall(".//thing"):
        audit["things"] += 1
        nodes = {}
        for node in thing.findall("./nodes/node"):
            ident = int(node.attrib["id"])
            xyz = np.array([float(node.attrib[a]) for a in "xyz"], dtype=np.float64)
            if ident in nodes or not np.isfinite(xyz).all():
                raise ValueError(f"Duplicate or invalid node {ident}")
            nodes[ident] = xyz
        audit["nodes"] += len(nodes)
        audit["empty_things"] += not bool(nodes)
        graph = {ident: set() for ident in nodes}
        for edge in thing.findall("./edges/edge"):
            a, b = int(edge.attrib["source"]), int(edge.attrib["target"])
            if a not in nodes or b not in nodes:
                raise ValueError("NML edge references an absent node")
            audit["duplicate_edges"] += b in graph[a]
            graph[a].add(b)
            graph[b].add(a)
        remaining = set(nodes)
        while remaining:
            todo = [min(remaining)]
            component = set()
            while todo:
                node = todo.pop()
                if node in component:
                    continue
                component.add(node)
                todo.extend(graph[node] - component)
            remaining.difference_update(component)
            audit["components"] += 1
            if len(component) < 2:
                audit["singleton_components"] += 1
                continue
            if any(len(graph[n]) > 2 for n in component):
                audit["branch_components"] += 1
                continue
            ends = sorted(n for n in component if len(graph[n]) == 1)
            if len(ends) != 2 or any(n in graph[n] for n in component):
                audit["cyclic_components"] += 1
                continue
            ordered, previous, current = [], None, ends[0]
            while current is not None:
                ordered.append(current)
                neighbors = graph[current] - ({previous} if previous is not None else set())
                previous, current = current, next(iter(neighbors), None)
            if len(ordered) != len(component):
                raise ValueError("Incomplete NML traversal")
            points = np.stack([nodes[n] for n in ordered])
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
            audit["zero_length_edges"] += int((~keep).sum())
            points = points[keep]
            if len(points) >= 2:
                paths.append(points)
                audit["valid_paths"] += 1
    return paths, dict(audit), spacing


def sample_paths(paths, origin_xyz, shape_zyx, *, spacing=8, half_window=8, border=32, audit=None):
    positions, tangents, ids = [], [], []
    counts = Counter(audit or {})
    counts.update({"short_paths": 0, "border_excluded_samples": 0,
                   "degenerate_tangent_samples": 0, "eligible_paths": 0})
    origin_xyz = np.asarray(origin_xyz)
    for ident, points in enumerate(paths):
        local = (points - origin_xyz)[:, ::-1]
        arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
        if arclength[-1] < 2 * half_window:
            counts["short_paths"] += 1
            continue
        query = np.arange(half_window, arclength[-1] - half_window + 1e-8, spacing)
        def interpolate(s):
            return np.stack([np.interp(s, arclength, local[:, axis]) for axis in range(3)], axis=1)
        p = interpolate(query)
        delta = interpolate(query + half_window) - interpolate(query - half_window)
        norm = np.linalg.norm(delta, axis=1)
        inside = ((p >= border) & (p <= np.asarray(shape_zyx) - 1 - border)).all(axis=1)
        valid = inside & (norm > 1e-6)
        counts["border_excluded_samples"] += int((~inside).sum())
        counts["degenerate_tangent_samples"] += int((inside & (norm <= 1e-6)).sum())
        if not valid.any():
            continue
        positions.append(p[valid])
        tangents.append(delta[valid] / norm[valid, None])
        ids.append(np.full(int(valid.sum()), ident, dtype=np.int32))
        counts["eligible_paths"] += 1
    if not positions:
        raise ValueError("No reference samples remain after exclusion")
    fiber_ids = np.concatenate(ids)
    count = np.bincount(fiber_ids)
    weights = 1.0 / count[fiber_ids]
    weights /= weights.sum()
    return Samples(np.concatenate(positions), np.concatenate(tangents),
                   fiber_ids, weights, dict(counts))


def sparse_tensor(image, positions_zyx, sigma, rhos, *, truncate=4):
    """Yield tensors at query locations without materializing dense eigenvectors.

    Gaussian derivatives form J = G_rho * (grad I grad I^T). Only six scalar
    fields are filtered, then trilinearly sampled. This is a reference method,
    not a drop-in claim of numerical equivalence to the Villa implementation.
    """
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        gradients.append(gaussian_filter(image, sigma, order=order, mode="reflect", truncate=truncate))
    del image
    tensors = {rho: np.empty((len(positions_zyx), 3, 3), dtype=np.float64) for rho in rhos}
    for i in range(3):
        for j in range(i, 3):
            product = gradients[i] * gradients[j]
            for rho in rhos:
                field = gaussian_filter(product, rho, mode="reflect", truncate=truncate)
                values = map_coordinates(field, positions_zyx.T, order=1, mode="nearest", prefilter=False)
                tensors[rho][:, i, j] = values
                tensors[rho][:, j, i] = values
                del field
            del product
    for rho in rhos:
        yield rho, tensors[rho]


def decompose(tensors):
    values, vectors = np.linalg.eigh(tensors)
    nonnegative = np.maximum(values, 0)
    confidence = ((nonnegative[:, 1] - nonnegative[:, 0]) /
                  (nonnegative[:, 1] + nonnegative[:, 0] + 1e-12))
    return vectors[:, :, 0], confidence, values


def angular_error(prediction, reference):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    pn, rn = np.linalg.norm(prediction, axis=-1), np.linalg.norm(reference, axis=-1)
    if np.any(pn < 1e-12) or np.any(rn < 1e-12):
        raise ValueError("Direction vectors cannot have zero norm")
    dot = np.sum(prediction * reference, axis=-1) / (pn * rn)
    return np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    return float(np.interp(q * cdf[-1], cdf, values[order]))


def metrics(errors, weights, accepted=None):
    accepted = np.ones(len(errors), dtype=bool) if accepted is None else accepted
    mass = float(weights[accepted].sum())
    out = {"samples": int(accepted.sum()), "coverage": mass / float(weights.sum())}
    if mass <= 0:
        return {**out, "mean_degrees": None, "median_degrees": None, "p90_degrees": None, "over_30_degrees": None}
    w, e = weights[accepted], errors[accepted]
    return {**out, "mean_degrees": float(np.dot(w, e) / mass),
            "median_degrees": weighted_quantile(e, w, 0.5),
            "p90_degrees": weighted_quantile(e, w, 0.9),
            "over_30_degrees": float(w[e > 30].sum() / mass)}

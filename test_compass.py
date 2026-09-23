import xml.etree.ElementTree as ET

import numpy as np
import pytest

from compass import read_paths, sample_paths, sparse_tensor, decompose, angular_error, metrics


def write_nml(tmp_path, nodes, edges, scale=(7.91, 7.91, 7.91)):
    root = ET.Element("things")
    parameters = ET.SubElement(root, "parameters")
    ET.SubElement(parameters, "scale", dict(zip("xyz", map(str, scale))))
    thing = ET.SubElement(root, "thing", {"id": "3"})
    ns = ET.SubElement(thing, "nodes")
    es = ET.SubElement(thing, "edges")
    for ident, xyz in nodes:
        ET.SubElement(ns, "node", {"id": str(ident), **dict(zip("xyz", map(str, xyz)))})
    for a, b in edges:
        ET.SubElement(es, "edge", {"source": str(a), "target": str(b)})
    path = tmp_path / "fixture.nml"
    ET.ElementTree(root).write(path)
    return path


def test_graph_order_and_xyz_to_zyx_with_nonzero_origin(tmp_path):
    file = write_nml(tmp_path, [(30, [160, 260, 380]), (2, [120, 260, 380]),
                                (90, [140, 260, 380])], [(90, 30), (2, 90)])
    paths, audit, _ = read_paths(file)
    np.testing.assert_array_equal(paths[0][:, 0], [120, 140, 160])
    samples = sample_paths(paths, [100, 200, 300], [128] * 3, spacing=8, half_window=8, border=10)
    np.testing.assert_allclose(samples.positions_zyx[:, 0], 80)
    np.testing.assert_allclose(samples.positions_zyx[:, 1], 60)
    np.testing.assert_allclose(samples.tangents_zyx, np.tile([0, 0, 1], (len(samples.fiber_ids), 1)))
    assert audit["valid_paths"] == 1


@pytest.mark.parametrize("edges,counter", [([(1, 2), (2, 3), (2, 4)], "branch_components"),
                                          ([(1, 2), (2, 3), (3, 4), (4, 1)], "cyclic_components")])
def test_ambiguous_graphs_are_counted_and_excluded(tmp_path, edges, counter):
    path = write_nml(tmp_path, [(i, [i, 0, 0]) for i in range(1, 5)], edges)
    paths, audit, _ = read_paths(path)
    assert not paths
    assert audit[counter] == 1


def test_invalid_metadata_and_edges_fail_loudly(tmp_path):
    path = write_nml(tmp_path, [(1, [0, 0, 0])], [(1, 2)])
    with pytest.raises(ValueError, match="absent node"):
        read_paths(path)
    path = write_nml(tmp_path, [], [], (1, 1, 2))
    with pytest.raises(ValueError, match="isotropic"):
        read_paths(path)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_known_straight_cylinder_has_correct_axis(axis):
    coordinates = np.indices((35, 35, 35), dtype=np.float32) - 17
    distance = sum(coordinates[i] ** 2 for i in range(3) if i != axis)
    image = (200 * np.exp(-distance / 12)).astype(np.float32)
    positions = np.array([[17, 17, 17], [16, 17, 18]], dtype=float)
    _, tensor = next(sparse_tensor(image, positions, 1, [2]))
    predicted, confidence, _ = decompose(tensor)
    target = np.zeros((2, 3))
    target[:, axis] = 1
    np.testing.assert_allclose(angular_error(predicted, target), 0, atol=1e-5)
    assert np.all(confidence > .99)


def test_axis_sign_and_nonunit_vectors():
    np.testing.assert_allclose(angular_error([[2, 0, 0], [-1, 0, 0], [0, 1, 0]],
                                           [[1, 0, 0]] * 3), [0, 0, 90])
    with pytest.raises(ValueError, match="zero norm"):
        angular_error([[0, 0, 0]], [[1, 0, 0]])


def test_fiber_weighting_is_independent_of_length():
    paths = [np.array([[20, 20, 20], [60, 20, 20]]),
             np.array([[20, 40, 20], [100, 40, 20]])]
    samples = sample_paths(paths, [0, 0, 0], [128] * 3, border=10)
    for ident in [0, 1]:
        np.testing.assert_allclose(samples.weights[samples.fiber_ids == ident].sum(), .5)
    errors = np.where(samples.fiber_ids == 0, 10., 40.)
    assert metrics(errors, samples.weights)["mean_degrees"] == pytest.approx(25)
    subset = metrics(errors, samples.weights, samples.fiber_ids == 0)
    assert subset["coverage"] == pytest.approx(.5)
    assert subset["over_30_degrees"] == 0


def test_no_confidence_on_constant_image():
    _, tensor = next(sparse_tensor(np.ones((19,) * 3), np.array([[9, 9, 9]]), 1, [1]))
    _, confidence, _ = decompose(tensor)
    np.testing.assert_allclose(confidence, 0)
    assert metrics(np.array([40.]), np.array([1.]), np.array([False]))["mean_degrees"] is None

import numpy as np
import pytest

from compass import sparse_tensor, decompose
from export_field import tensor_blocks


def test_tiling_matches_full_volume_including_seams_and_edges():
    image = np.random.default_rng(47).integers(0, 256, (37, 41, 39), dtype=np.uint8)
    stride = 3
    grid = np.stack(np.meshgrid(*(np.arange(0, n, stride) for n in image.shape), indexing="ij"), axis=-1)
    _, tensor = next(sparse_tensor(image, grid.reshape(-1, 3), 1.5, [3]))
    axes, conf, eigenvalues = decompose(tensor)
    expected = (axes[:, :, None] * axes[:, None, :])[:, *np.triu_indices(3)].reshape(*grid.shape[:-1], 6)
    actual = np.zeros_like(expected)
    actual_conf = np.zeros(grid.shape[:-1])
    visits = np.zeros(grid.shape[:-1], dtype=int)
    for selection, packed, c, values in tensor_blocks(image, 1.5, 3, stride=stride, block=18):
        actual[selection] = packed
        actual_conf[selection] = c
        visits[selection] += 1
    np.testing.assert_array_equal(visits, 1)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(actual_conf, conf.reshape(grid.shape[:-1]), atol=1e-6)


def test_invalid_block_cannot_duplicate_or_skip_grid_points():
    with pytest.raises(ValueError, match="multiple"):
        next(tensor_blocks(np.zeros((30,) * 3), 1, 2, stride=4, block=17))

import numpy as np
import pytest
import zarr

from sample_field import sample


def field(tmp_path):
    group = zarr.open_group(str(tmp_path / "field.zarr"), mode="w", zarr_format=2)
    group.attrs.update({"format": "fiber-compass-1", "complete": True, "axes": "ZYX",
                        "projector_components": ["zz", "zy", "zx", "yy", "yx", "xx"],
                        "source_origin_xyz": [100, 200, 300], "grid_stride_source_voxels": 4})
    p = np.zeros((3, 3, 3, 6), dtype=np.float32)
    p[..., 0] = 1
    group.create_array("axis_projector", data=p, chunks=(2, 2, 2, 6))
    group.create_array("confidence", data=np.full((3, 3, 3), .8, dtype=np.float32), chunks=(2, 2, 2))
    return group


def test_origin_transform_chunk_seams_and_missing_points(tmp_path):
    group = field(tmp_path)
    # Reference origin is one source voxel earlier in Z, same X/Y.
    p = np.array([[1, 0, 0], [7, 7, 7], [9, 8, 8], [10, 0, 0]], dtype=float)
    axes, confidence = sample(group, p, [100, 200, 299])
    np.testing.assert_allclose(np.abs(axes[:3]), np.tile([1, 0, 0], (3, 1)))
    np.testing.assert_allclose(confidence[:3], .8)
    assert np.isnan(axes[3]).all() and confidence[3] == 0


def test_conflicting_axes_lose_interpolation_confidence(tmp_path):
    group = field(tmp_path)
    p = group["axis_projector"][:]
    p[1:, ..., 0] = 0
    p[1:, ..., 3] = 1
    group["axis_projector"][:] = p
    _, confidence = sample(group, np.array([[2, 0, 0]]), [100, 200, 300])
    np.testing.assert_allclose(confidence, 0, atol=1e-6)


def test_partial_export_is_not_accepted(tmp_path):
    group = field(tmp_path)
    group.attrs["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        sample(group, np.zeros((1, 3)), [0, 0, 0])

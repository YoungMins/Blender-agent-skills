"""Guard: the calibrated views of the ``hatted`` benchmark case agree with each other and with the ground truth in world space.

``hatted`` holds a staff that drags the silhouette bbox off the body axis, so the front / back origin moves to the body axis
(a pure X translation of the model). A change that moved front and back differently, moved the side view, or broke the
baseline would show here as a world-space disagreement (benchmark regression of 2026-10-23: held-out IoU 87 -> 71)."""
import pytest

from bbr.core import geomscore
from bbr.core.imageops import mask_bbox
from bbr.core.measure import load_view_data, load_calibrations
from bbr.core.config import PipelineConfig
from bbr.core.conventions import view_frame
from conftest import SK

CASE = SK / "bench" / "cases" / "hatted"
TOL_M = 0.002

pytestmark = pytest.mark.skipif(not (CASE / "gt.obj").is_file(), reason="hatted benchmark case not available")


@pytest.fixture(scope="module")
def frame():
    spec, calibs, info = load_calibrations(CASE / "view_spec.json")
    _, masks = load_view_data(spec, PipelineConfig())
    verts, _ = geomscore.read_obj(CASE / "gt.obj")
    return calibs, info, masks, verts


def _residual_m(calib, mask, verts):
    """(horizontal, vertical) world offset of the real silhouette from where the calibration puts the ground-truth mesh."""
    uv = calib.world_to_px(verts)
    x0, y0, x1, y1 = mask_bbox(mask)
    f = view_frame(calib.camera_axis)
    sign = 1.0 if f.right_axis[0] == "+" else -1.0
    horizontal = sign * ((x0 + x1) / 2.0 - (uv[:, 0].min() + uv[:, 0].max()) / 2.0) * calib.m_per_px
    vertical = -(y1 - uv[:, 1].max()) * calib.m_per_px      # baseline row vs the projected ground-truth floor
    return horizontal, vertical


def test_front_and_back_origins_agree_in_world_space(frame):
    calibs, _, masks, verts = frame
    front, back = _residual_m(calibs["front"], masks["front"], verts), _residual_m(calibs["back"], masks["back"], verts)
    assert abs(front[0] - back[0]) <= TOL_M, (front, back)


def test_baselines_agree_with_the_ground_truth_floor(frame):
    calibs, _, masks, verts = frame
    for view in ("front", "back", "side"):
        assert abs(_residual_m(calibs[view], masks[view], verts)[1]) <= TOL_M, view


def test_side_origin_is_the_bbox_centre(frame):
    calibs, _, masks, verts = frame
    assert abs(_residual_m(calibs["side"], masks["side"], verts)[0]) <= TOL_M


def test_reported_frame_offset_is_the_front_back_translation(frame):
    """What the benchmark subtracts (``frame_offset_m``) is exactly the world shift of the front / back origin."""
    calibs, info, masks, verts = frame
    shift = _residual_m(calibs["front"], masks["front"], verts)[0]
    assert abs(shift - info["frame_offset_m"]["x"]) <= TOL_M, (shift, info["frame_offset_m"])
    assert abs(info["frame_offset_m"].get("y", 0.0)) <= TOL_M

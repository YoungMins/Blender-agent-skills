"""Image load/save. PIL when importable, else bpy (imported lazily inside functions).

All arrays are float32 ``H x W x 4`` RGBA in [0, 1], TOP-DOWN (row 0 = top). bpy pixels are
bottom-up and are converted here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import ImageError, get_logger

log = get_logger(__name__)

_BYTE_MAX = 255.0


def _have_pil() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def load_rgba(path: Path | str) -> np.ndarray:
    """Load an image as float32 HxWx4 RGBA, top-down. Raises ImageError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"image not found: {path}")
    if _have_pil():
        from PIL import Image
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / _BYTE_MAX
        except Exception as exc:  # PIL raises many types
            raise ImageError(f"cannot read image {path}: {exc}") from exc
        return np.ascontiguousarray(arr)
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise ImageError("neither PIL nor bpy is available to read images") from exc
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"  # no sRGB->linear conversion of pixel values
        img.alpha_mode = "STRAIGHT"
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return np.ascontiguousarray(buf.reshape(h, w, 4)[::-1])
    finally:
        bpy.data.images.remove(img)


def save_rgba(path: Path | str, rgba: np.ndarray) -> None:
    """Save HxWx3/4 float image (top-down, [0,1]) as PNG (format from extension with PIL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,), np.float32)], axis=2)
    if _have_pil():
        from PIL import Image
        Image.fromarray(np.rint(np.clip(arr, 0, 1) * _BYTE_MAX).astype(np.uint8), "RGBA").save(path)
        return
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise ImageError("neither PIL nor bpy is available to write images") from exc
    h, w = arr.shape[:2]
    img = bpy.data.images.new("bbr_tmp_save", w, h, alpha=True)
    try:
        img.colorspace_settings.name = "Non-Color"
        img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).ravel())
        img.filepath_raw = str(path)
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def save_mask(path: Path | str, mask: np.ndarray) -> None:
    """Save a boolean mask as a black/white PNG."""
    save_rgba(path, np.asarray(mask, dtype=np.float32))


def load_view_image(view) -> np.ndarray:
    """Load a ViewEntry's image with ``crop_px`` applied first, then ``flip_horizontal``."""
    rgba = load_rgba(view.path)
    if view.crop_px:
        x0, y0, x1, y1 = view.crop_px
        h, w = rgba.shape[:2]
        if x1 > w or y1 > h:
            raise ImageError(f"crop_px {view.crop_px} exceeds image size {(w, h)} of {view.path}")
        rgba = rgba[y0:y1, x0:x1]
    if view.flip_horizontal:
        rgba = rgba[:, ::-1]
    return np.ascontiguousarray(rgba)

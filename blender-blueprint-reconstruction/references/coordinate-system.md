# Coordinate System

Single source of truth in code: `scripts/bbr/core/conventions.py`. If this document and the code ever disagree, the code wins and this document is a bug.

## World

- Blender world, meters, **Z up**.
- The object's **front faces -Y**. The object's own right side is at **-X**. A character facing you in the front view has its right hand on the image's left.
- **Origin** is the footprint center on the floor: the X and Y bbox centers are 0, and the lowest point is at Z = 0.

## Views

`camera_axis` is the world direction **from the origin to the camera**. Image coordinates: `u` points right and `v` points **down** (row 0 is the top row).

| view key | camera_axis | camera looks along | image right | image up | Blender numpad |
|---|---|---|---|---|---|
| `front`  | `-Y` | +Y | +X | +Z | 1 |
| `back`   | `+Y` | -Y | -X | +Z | Ctrl+1 |
| `right`  | `+X` | -X | +Y | +Z | 3 |
| `left`   | `-X` | +X | -Y | +Z | Ctrl+3 |
| `top`    | `+Z` | -Z | +X | +Y | 7 |
| `bottom` | `-Z` | +Z | +X | -Y | Ctrl+7 |

`side` is allowed as a view key, but it **must** state `camera_axis` explicitly.

### Rule for profile images

> In a profile image, if the object's front (face, nose, visor, drawer fronts) points to **image-right**, use `camera_axis: "-X"`. If it points to **image-left**, use `"+X"`.

Getting this wrong mirrors the model front-to-back. Front and side IoU **cannot detect this**, because the validation camera uses the same wrong convention. That is why the pipeline always renders a 3/4 perspective preview (`previews/perspective.png`), and why you must look at it before accepting a result.

`flip_horizontal: true` mirrors an image before use. Use it for references that are known to be mirrored.

## Left and right (`_l` / `_r` names)

`_l` / `_r` always mean the CHARACTER's own left / right, never the image's. The character faces -Y, so its left side is +X, its right side -X (the rule above: its right hand is on the image's left in the front view). Name every part pair by this, in labels, component ids and `mirror_of`.

```
 top view (X right, Y up)            front view, camera -Y         back view, camera +Y
                                      image-left | image-right      image-left | image-right
        +Y  back                      _r  (-X)   |  _l  (+X)        _l  (+X)   |  _r  (-X)
         ^
 _r -X <-+-> +X _l                   side view, camera -X: the character's RIGHT side (_r) faces the camera
         v                           (character faces image-right); camera +X shows its left side (_l)
        -Y  character faces this way
```

- Front view: image-LEFT = the character's right (`_r`, -X), image-RIGHT = its left (`_l`, +X).
- Back view: mirrored: image-LEFT = `_l` (+X), image-RIGHT = `_r` (-X).
- Side view with `camera_axis` `-X` (character faces image-right): you see the character's right side; `+X` (faces image-left) shows its left side.
- `prepare_references.py labels` and `anchors` warn when a `_l` / `_r` label's centroid sits on the wrong side of X = 0 in a front / back view (centroid x vs the name, dead zone 2% of the height). The usual cause is naming by the image side: ear at image-left in the front view is `ear_r`.

## Calibration (`silhouette_bbox` mode)

1. A mask is extracted for every view.
2. `resolve_dimensions()` combines the known dimensions with the silhouette aspect ratios: front and back give W/H, the sides give D/H, and top gives W/D. Each resolved dimension records its provenance (`known`, `derived:<view>`, or `default`).
3. The scale (m/px) comes from the view's vertical extent for Z-up views, or its horizontal extent for top and bottom.
4. Origin pixel: horizontally, the bbox center. Vertically, the bbox bottom (the floor) for Z-up views, or the bbox center for top and bottom.

Reference planes, validation cameras and comparison renders all derive from the **same** `ViewCalibration`. The render pixel grid therefore equals the reference pixel grid, so IoU measures geometry and not framing.

## Collections

| Collection | Contents | Rendered in validation |
|---|---|---|
| `REF_BLUEPRINT` | Reference image planes (unselectable) | no |
| `AI_MODEL` | Generated model components and `AI_ORIGIN` | yes |
| `AI_DEBUG` | Cameras, bounding volume, debug helpers | no |

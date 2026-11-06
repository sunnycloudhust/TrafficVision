# Data labelling tools

Run the calibration GUI from the repository root:

```bash
python data/data-labelling/calibration_tool.py
```

Select a camera, click at least four non-collinear points on the road, then
choose a geometry mode. For a rectangle, click its four corners in order around
the shape and enter its length and width in meters. For another shape, enter
the real distance in meters for every pair of selected points. The image
is shown with a 25% margin around it; clicking there records negative pixel
coordinates or coordinates beyond the frame dimensions. Use these only for
known geometric points whose projected image coordinates can be established.
In the other-shape mode, the tool uses pairwise distances to construct 2D world
coordinates. It anchors point 1 at `(0, 0)` and point 2 on the positive X axis.
Make sure selected points lie on one flat road plane and measurements are
accurate.

Save writes matching `image_points_<CAMERA>.json` and
`world_points_<CAMERA>.json` files under `data/ua_detrac/world_points/`. These
can be passed to `scripts/calibrate_camera.py` in matching point order.

The zone drawing tool is available with:

```bash
python data/data-labelling/zone_labeler.py
```

It lists camera sequences and opens each sequence's first available image as a
representative frame. Saved zone configurations are marked `--done`.

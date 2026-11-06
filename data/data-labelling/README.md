# Data labelling tools

Run the calibration GUI from the repository root:

```bash
python data/data-labelling/calibration_tool.py
```

For a different archive layout or a temporary output folder:

```bash
python data/data-labelling/calibration_tool.py \
  --frames-dir /path/to/sequences --output-dir /path/to/calibration-points
```

Each sequence folder must contain image frames. Install Tk for the selected
Python interpreter and use a graphical session (WSLg or desktop). Geometry
validation itself can be tested without a display.

Select a camera, click at least four non-collinear points on the road, then
choose a geometry mode. For a rectangle, click its four corners first in order
around the shape and enter its length and width in meters. Each extra point only
needs distances to the first three corners. For another shape, enter the real
distance in meters for every pair of selected points. The image
is shown with a 25% margin around it. Clicking outside the image is disabled
by default. Enable the extrapolation checkbox only for known geometric points
whose projected image coordinates have been established independently. The
margin does not infer the road position or provide extra observations.
In the other-shape mode, the tool uses pairwise distances to construct 2D world
coordinates. It anchors point 1 at `(0, 0)` and point 2 on the positive X axis.
Make sure selected points lie on one flat road plane and measurements are
accurate.

The tool rejects distance reconstructions and homography fits outside the
declared error tolerance (default 0.25 metres). The first four rectangle corners
must form a convex quadrilateral in clockwise or counterclockwise order.
Conflicting optional measurements are checked too. Editing dimensions or the
error tolerance invalidates the previous solution; calculate again before save.

This tolerance is an input consistency check, not an accuracy certificate.
Use a real rectangle on the road; an arbitrary image quadrilateral does not
become a measured rectangle by entering a length/width. In near-horizontal
views choose visible, well-spread road points and validate the result on
independent measured points. A four-point fit can be perfect despite incorrect
physical dimensions. This GUI does not estimate metres from pixels alone.

Save writes matching `image_points_<CAMERA>.json` and
`world_points_<CAMERA>.json` files under `data/ua_detrac/world_points/`. These
can be passed to `scripts/calibrate_camera.py` in matching point order.
An additional `calibration_points_<CAMERA>.json` records input dimensions,
distance measurements, tolerance, frame reference and extrapolated point IDs.
The `--done` label means saved point pairs pass numerical validation; it does
not confirm the physical measurements or speed accuracy.

Generate the camera YAML after saving the point pairs:

```bash
python scripts/calibrate_camera.py --image /path/to/frame.jpg \
  --image-points /path/to/calibration-points/image_points_MVI_20011.json \
  --world-points /path/to/calibration-points/world_points_MVI_20011.json \
  --camera-id MVI_20011 --fps 25 --output configs/cameras/MVI_20011.yaml
```

The zone drawing tool is available with:

```bash
python data/data-labelling/zone_labeler.py
```

It lists camera sequences and opens each sequence's first available image as a
representative frame. Saved zone configurations are marked `--done`.

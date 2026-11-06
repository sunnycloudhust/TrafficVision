# Data labelling

## Create polygon zones

Run the GUI from the repository root:

```bash
python data-labelling/zone_labeler.py \
  --source /path/to/MVI_20011.mp4 \
  --output configs/zones/MVI_20011.yaml
```

Controls:

- Left click: add a polygon point
- `Enter`: finish the current polygon and enter its name
- `U`: undo the last point
- `C`: clear the current polygon
- `S`: save the YAML file
- `Q` or `Esc`: quit

The generated file includes the source frame size and is compatible with
`--zone-config` in the TrafficVision demo command. Create one output YAML per
video/camera.

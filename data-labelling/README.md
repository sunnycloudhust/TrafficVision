# Data labelling

## Create polygon zones

Run the GUI from the repository root. It lists every camera sequence under
`data/ua_detrac/raw/DETRAC/DETRAC-Images/DETRAC-Images`, opens the first image
in the selected sequence as its representative frame, and marks cameras with
saved zones as `--done`:

```bash
python data-labelling/zone_labeler.py
```

Controls:

- Left click: add a polygon point
- `Enter`: finish the current polygon and enter its name
- `U`: undo the last point
- `C`: clear the current polygon
- `S`: save the YAML file
- `Q` or `Esc`: quit

The generated file includes the source frame size and is compatible with
`--zone-config` in the TrafficVision demo command. It creates one output YAML
per camera.

# Hugging Face source synchronization

This branch syncs source, configuration, scripts, camera calibration points, and the small dataset manifests from `sunnycloudhust/TrafficVision` at commit `4af5adbead7219881a9c8c6a7dfa06c454d9387c`. Existing GitHub-only files are retained. The GitHub `.gitignore` is retained to exclude datasets, model weights and generated outputs. The report includes small evidence files for its reproducibility audit; see `report/README.md`.

The report describes artifacts from this snapshot; neither calibration accuracy nor experimental comparability is implied by inclusion. Large datasets and checkpoints remain on Hugging Face and must be obtained separately when needed.

## Calibration review

All 100 `configs/cameras/MVI_*.yaml` files in the imported snapshot have non-null homographies; on GitHub main, many of these were null templates. Their inclusion does not establish that world-point distances were physically measured or independently validated. Check each camera's image points, measured road dimensions, frame geometry, and held-out validation before treating its speed estimates as metric ground truth. The imported zones also replace some existing zone definitions; review changes for each affected camera before merging.

# Dataset notes

The repository currently contains UA-DETRAC images and XML annotations under `data/DETRAC-dataset/`. The runnable defaults are in `configs/dataset.yaml`. `data/raw/` and `data/processed/` are reserved for user-provided datasets and generated indexes; large dataset files are ignored by Git.

UA-DETRAC XML supplies sequence/frame numbers, bounding boxes, `vehicle_type`, and per-sequence target IDs. Frames from one sequence must stay in the same split. The training workflow therefore divides the official training sequences into train and validation sets using the configured seed; the official test annotations remain held out.

For another dataset, use COCO JSON (`images`, `annotations`, `categories` with `image_id`, `bbox` in `[x,y,width,height]`, and `category_id`) and configure image root, classes, and category mapping. Class names and paths must match the dataset; label 0 is reserved for background internally.

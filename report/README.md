# TrafficVision report

English LaTeX report following the Elsevier `elsarticle` structure of the supplied template. The report is based on the files inspected in the Hugging Face dataset repository on 27 November 2026; it does not depend on downloading the video dataset or model checkpoints. The cited repository snapshot is `4af5adbead7219881a9c8c6a7dfa06c454d9387c`.

## Build

From this folder, with LaTeX installed (a local copy of `elsarticle.cls`, generated from the official CTAN source, is included):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` includes the files in `sections/`. The training-curve figure was obtained from `outputs/evaluation/training_loss_and_map_curves.png` in the project repository. All results are traced to published files in the references.

The small `evidence/` directory contains the original per-epoch logs, plotting script, aggregate CSV and detector comparison CSV used for the audit; it contains no images or checkpoints. Run `python evidence/check_curves.py` to verify the 95 aggregate rows against the source logs (within five-decimal rounding). This checks CSV provenance, not scientific validity or identity between validation subsets. The right-hand plot combines different sample counts and missing classes, as documented in the report.

## Before submission

- Replace `TrafficVision Project Team` and `Deep Learning Course Project` in `sections/FrontMatter.tex` with the confirmed author list, affiliations, course and lecturer. Names from the unrelated template were deliberately not copied.
- Ask the team to confirm the GPU model, software environment, exact evaluated frame identities, sampling strategy and any newer results. Add held-out test, tracking and speed metrics only when backed by the corresponding artifacts and protocol.
- Check whether the team has added newer results after the cited snapshot; update the evidence and pin a new revision if needed.
- Verify that all calibration dimensions are measured and validate on independent points before describing metric speed as accurate.

The provided `report.tex` in the parent workspace is a ZIP of the original example project despite its extension. It is retained as a reference and is not part of this report.

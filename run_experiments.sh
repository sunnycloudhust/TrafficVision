#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python main.py --command train --model faster_rcnn --initialization pretrained
python main.py --command evaluate --model faster_rcnn --initialization pretrained

# python main.py --command train --model faster_rcnn --initialization scratch
# python main.py --command evaluate --model faster_rcnn --initialization scratch

# python main.py --command train --model fast_rcnn --initialization pretrained
# python main.py --command evaluate --model fast_rcnn --initialization pretrained

# python main.py --command train --model rcnn --initialization pretrained
# python main.py --command evaluate --model rcnn --initialization pretrained

python main.py --command track-all

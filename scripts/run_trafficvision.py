"""Run the TrafficVision CLI directly from a src-layout checkout."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from trafficvision.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

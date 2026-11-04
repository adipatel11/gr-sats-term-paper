import os
from pathlib import Path

# code/grsats/config.py -> project root is 2 folders up
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
# on colab this gets pointed at google drive so outputs survive a disconnect
OUT = Path(os.environ.get("GRSATS_OUT", ROOT / "code" / "outputs"))

SEEDS = [0, 1, 2]
DEV_FRAC = 0.1                  # part of train held out to pick the probe strength
L2_GRID = [0.01, 0.1, 1, 10, 100, 1000]  # l2 strength, sklearn's C is 1 / strength
BATCH_SIZE = 64

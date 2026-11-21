import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# on colab this is set to a drive folder so outputs survive a disconnect
OUT = Path(os.environ.get("GRSATS_OUT", ROOT / "code" / "outputs"))

SEEDS = [0, 1, 2]
DEV_FRAC = 0.1
BATCH_SIZE = 64
L2_GRID = [0.01, 0.1, 1, 10, 100, 1000]

# lora settings from the paper
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 3e-4
LORA_WD = 0.01
LORA_EPOCHS = 20
LORA_PATIENCE = 5

from grsats.data.ham10000 import load_ham
from grsats.data.fitzpatrick17k import load_fitz

LOADERS = {"ham": load_ham, "fitz": load_fitz}
GROUPS = {"ham": ["sex", "age"], "fitz": ["skin"]}  # group columns for each dataset

# prints images (malignant, % malignant) per split and per group in the pool.
# should match the data table in the paper. run from code/: python -m scripts.check_data
import os

from grsats.data import GROUPS, LOADERS


def count(d):
    return f"{len(d):,} ({d['y'].sum():,}, {d['y'].mean():.1%})"


for name, load in LOADERS.items():
    df = load()
    pool = df[df["split"].isin(["val", "test"])]

    print(f"\n{name}: {len(df):,} images")
    for s in ["train", "val", "test"]:
        print(f"  {s:<12} {count(df[df['split'] == s])}")
    print(f"  {'pool':<12} {count(pool)}")
    for col in GROUPS[name]:
        for g, d in pool.groupby(col):
            print(f"  pool {g:<7} {count(d)}")

    missing = sum(not os.path.exists(p) for p in df["path"])
    print(f"  missing image files: {missing}")

# prints images (malignant, % malignant) per split and per group in the pool
# these should match the data table in the paper
from data import GROUPS, LOADERS, missing_images


def count(d):
    return f"{len(d):,} ({d['y'].sum():,}, {d['y'].mean():.1%})"


for name, load in LOADERS.items():
    df = load()
    pool = df[df["split"].isin(["val", "test"])]

    print(f"\n{name}: {len(df):,} images")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {count(df[df['split'] == s])}")
    print(f"  pool: {count(pool)}")
    for col in GROUPS[name]:
        for g, d in pool.groupby(col):
            print(f"  pool {g}: {count(d)}")
    print(f"  missing image files: {missing_images(df)}")

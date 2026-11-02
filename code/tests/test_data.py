# run from code/: python -m pytest tests
import pytest

from grsats.data import GROUPS, LOADERS
from grsats.data.splits import add_dev

# (images, malignant) from the data table in the paper. groups are counted in the pool (val + test)
EXPECTED = {
    "ham": {"train": (8164, 1740), "val": (568, 63), "test": (1216, 149),
            "female": (820, 80), "male": (964, 132), "<60": (1206, 62), "60+": (578, 150)},
    "fitz": {"train": (7779, 1144), "val": (1100, 154), "test": (2219, 325),
             "I-II": (1794, 287), "III-IV": (1163, 160), "V-VI": (362, 32)},
}


@pytest.mark.parametrize("name", ["ham", "fitz"])
def test_counts_match_paper(name):
    df = LOADERS[name]()
    pool = df[df["split"].isin(["val", "test"])]
    got = dict(list(df.groupby("split")))
    for col in GROUPS[name]:
        got.update(dict(list(pool.groupby(col))))
    for key, expected in EXPECTED[name].items():
        assert (len(got[key]), got[key]["y"].sum()) == expected, key


@pytest.mark.parametrize("name", ["ham", "fitz"])
def test_no_missing_values(name):
    df = LOADERS[name]()
    assert df["id"].is_unique
    assert df[["y", "split"] + GROUPS[name]].notna().all().all()


@pytest.mark.parametrize("name", ["ham", "fitz"])
def test_each_lesion_in_one_split(name):
    df = add_dev(LOADERS[name](), seed=0)
    assert df.groupby("lesion")["split"].nunique().max() == 1


def test_dev_slice():
    df = LOADERS["ham"]()
    a, b = add_dev(df, 0), add_dev(df, 1)
    frac = (a["split"] == "dev").sum() / (df["split"] == "train").sum()
    assert 0.08 < frac < 0.12
    assert a["split"].equals(add_dev(df, 0)["split"])  # same seed, same dev slice
    assert not a["split"].equals(b["split"])           # new seed, new dev slice

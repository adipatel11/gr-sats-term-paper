# python -m pytest -q
import torch
from torch import nn

from data import add_dev, load_fitz, load_ham
from models import LoRAqkv


def count(d):
    return len(d), d["y"].sum()


# (images, malignant) from the data table in the paper, groups are counted in the pool
def test_ham_counts():
    df = load_ham()
    pool = df[df["split"].isin(["val", "test"])]
    assert count(df[df["split"] == "train"]) == (8164, 1740)
    assert count(df[df["split"] == "val"]) == (568, 63)
    assert count(df[df["split"] == "test"]) == (1216, 149)
    assert count(pool[pool["sex"] == "female"]) == (820, 80)
    assert count(pool[pool["sex"] == "male"]) == (964, 132)
    assert count(pool[pool["age"] == "<60"]) == (1206, 62)
    assert count(pool[pool["age"] == "60+"]) == (578, 150)


def test_fitz_counts():
    df = load_fitz()
    pool = df[df["split"].isin(["val", "test"])]
    assert count(df[df["split"] == "train"]) == (7779, 1144)
    assert count(df[df["split"] == "val"]) == (1100, 154)
    assert count(df[df["split"] == "test"]) == (2219, 325)
    assert count(pool[pool["skin"] == "I-II"]) == (1794, 287)
    assert count(pool[pool["skin"] == "III-IV"]) == (1163, 160)
    assert count(pool[pool["skin"] == "V-VI"]) == (362, 32)


def test_no_missing_values():
    for df in [load_ham(), load_fitz()]:
        assert df["id"].is_unique
        assert df.notna().all().all()


def test_each_lesion_in_one_split():
    assert add_dev(load_ham(), 0).groupby("lesion")["split"].nunique().max() == 1
    assert add_dev(load_fitz(), 0).groupby("lesion")["split"].nunique().max() == 1


def test_dev_slice():
    df = load_ham()
    a, b = add_dev(df, 0), add_dev(df, 1)
    assert 0.08 < (a["split"] == "dev").sum() / (df["split"] == "train").sum() < 0.12
    assert a["split"].equals(add_dev(df, 0)["split"])  # same seed, same dev
    assert not a["split"].equals(b["split"])           # new seed, new dev


def test_lora_starts_as_frozen_layer():
    qkv = nn.Linear(16, 48)
    x = torch.randn(2, 5, 16)
    assert torch.equal(LoRAqkv(qkv, rank=4, alpha=8)(x), qkv(x))


def test_lora_only_changes_q_and_v():
    qkv = nn.Linear(16, 48)
    lora = LoRAqkv(qkv, rank=4, alpha=8)
    nn.init.ones_(lora.b_q.weight)
    nn.init.ones_(lora.b_v.weight)
    x = torch.randn(2, 5, 16)
    diff = (lora(x) - qkv(x)).abs()
    assert diff[..., :16].sum() > 0     # q moved
    assert diff[..., 16:32].sum() == 0  # k did not
    assert diff[..., 32:].sum() > 0     # v moved

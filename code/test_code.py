# python -m pytest -q
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from torch import nn

from calibrate import atypicality, fit, fit_dac, make_folds, positivity, weights
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


def fake_logits(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(2, size=n).astype(float)
    z = 3 * (2 * y - 1) + 2 * rng.standard_normal(n)  # overconfident logits
    return z, y


def test_global_fit_is_temperature_scaling():
    z, y = fake_logits()
    sev = np.zeros(len(z), int)
    beta = fit(z[:, None], y, weights("clean", sev, []), positivity("global", 0, 0))[0]
    nll = lambda b: np.mean(np.logaddexp(0, b * z) - y * b * z)
    assert abs(beta - minimize_scalar(nll, bounds=(0.01, 10), method="bounded").x) < 1e-3


def test_shift_fit_never_flips_predictions():
    z, y = fake_logits()
    s = np.linspace(-2, 8, len(z))
    sev = np.repeat(np.arange(6), len(z) // 6 + 1)[:len(z)]
    theta = fit(np.stack([z, s * z], 1), y, weights("avg", sev, []), positivity("shift", s.min(), s.max()))
    beta = theta[0] + theta[1] * s
    assert (beta >= 0.05 - 1e-6).all()
    assert ((beta * z > 0) == (z > 0)).all()


def test_folds_keep_lesions_together():
    df = load_ham()
    dp = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    folds = make_folds(dp, "ham")
    assert set(folds) == {0, 1, 2, 3, 4}
    assert dp.assign(fold=folds).groupby("lesion")["fold"].nunique().max() == 1


def test_aar_two_class_form():
    # aar's softmax form with S_0, S_1 equals sigmoid(phi * z + S_1 - S_0) for a binary model
    rng = np.random.default_rng(0)
    z, phi, s0, s1 = rng.normal(size=4)
    logp = np.array([np.log(1 - expit(z)), np.log(expit(z))])
    scores = phi * logp + np.array([s0, s1])
    softmax = np.exp(scores[1]) / np.exp(scores).sum()
    assert abs(softmax - expit(phi * z + s1 - s0)) < 1e-12


def test_dac_with_constant_distance_is_temperature_scaling():
    z, y = fake_logits()
    w = np.full(len(z), 1 / len(z))
    predict, p = fit_dac(z, np.ones(len(z)), y, w)
    T = p[0] + p[1]
    brier = lambda t: np.mean((expit(z / t) - y) ** 2)
    assert abs(T - minimize_scalar(brier, bounds=(0.1, 10), method="bounded").x) < 1e-2


def test_atypical_points_score_higher():
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(500, 8))
    bank_y = rng.integers(2, size=500)
    a = atypicality(bank, bank_y, np.stack([np.zeros(8), np.full(8, 5.0)]))
    assert a[1] > a[0]

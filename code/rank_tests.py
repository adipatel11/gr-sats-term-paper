# tests across the 8 settings, runs locally on the csvs in results/:
# 1. friedman test + nemenyi critical difference on worst-group nll (like medfair)
# 2. h2: first severity where clean-fit ts is worse than no calibration, lp vs lora
# 3. the paper's success test: grsats vs each primary comparison averaged over all 8 settings, with a
#    paired lesion bootstrap (needs the *_perimage.npz files from calibrate_extra.py)
# python rank_tests.py
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

from config import N_BOOT

RESULTS = Path(__file__).resolve().parent / "results"
SETTINGS = [(ds, kind) for ds in ["ham", "fitz"] for kind in ["frozen", "lora"]]
MAIN = ["none", "ts_clean", "ts_gauss", "ts_corr", "dac", "md_ts", "aar", "cluster_ts", "grsats"]
PRIMARY = ["ts_corr", "dac", "aar"]


def load():
    # the extra csv has every method, so use it when it's there
    tabs = []
    for ds, kind in SETTINGS:
        f = RESULTS / f"{ds}_{kind}_extra.csv"
        if not f.exists():
            f = RESULTS / f"{ds}_{kind}_calibration.csv"
        tabs.append(pd.read_csv(f).assign(dataset=ds))
        print(f"using {f.name}")
    return pd.concat(tabs)


def friedman(res):
    methods = [m for m in MAIN if m in set(res["method"])]
    # one block per setting: worst-group nll averaged over the 3 seeds
    wide = res[res["method"].isin(methods)].groupby(["dataset", "model", "method"])["wg_nll"].mean().unstack()[methods]
    stat, p = friedmanchisquare(*[wide[m] for m in methods])
    ranks = wide.rank(axis=1).mean().sort_values()
    k, n = len(methods), len(wide)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    print(f"\nfriedman over {n} settings and {k} methods: chi2 = {stat:.2f}, p = {p:.2g}")
    print(f"nemenyi critical difference (alpha 0.05): {cd:.2f}. mean rank (1 = best):")
    print(ranks.round(2).to_string())
    return ranks, stat, p, cd


def h2(res):
    # severity 0 is clean. "harm" = clean-fit ts has higher worst-group nll than no calibration
    sev = [f"wg_nll sev{l}" for l in range(6)]
    mean = res.groupby(["dataset", "model", "method"])[sev].mean()
    print("\nh2: first severity where ts_clean is worse than none (worst-group nll, mean of 3 seeds)")
    for (ds, model), g in mean.groupby(level=[0, 1]):
        diff = g.xs("ts_clean", level=2).to_numpy()[0] - g.xs("none", level=2).to_numpy()[0]
        first = next((l for l in range(6) if diff[l] > 0), None)
        print(f"  {ds} {model:15s} first harm at severity {first}, ts_clean - none by severity: {np.round(diff, 3)}")


def pooled_bootstrap():
    # every image has the same rows in every setting, so the same lesion draws are used for all the
    # settings of a dataset. the two datasets have different images, so they get their own draws
    files = {(ds, kind): RESULTS / f"{ds}_{kind}_perimage.npz" for ds, kind in SETTINGS}
    if not all(f.exists() for f in files.values()):
        print("\nno *_perimage.npz in results/ yet, skipping the pooled bootstrap")
        return
    rng = np.random.default_rng(0)
    diffs = {(o, m): [] for o in PRIMARY for m in ["nll", "brier"]}
    point = {(o, m): [] for o in PRIMARY for m in ["nll", "brier"]}
    for ds in ["ham", "fitz"]:
        f0 = np.load(files[(ds, "frozen")])
        _, lid = np.unique(f0["lesion"], return_inverse=True)
        draws = rng.integers(lid.max() + 1, size=(N_BOOT, lid.max() + 1))
        counts = np.stack([np.bincount(d, minlength=lid.max() + 1) for d in draws])[:, lid]
        masks = [f0[k] for k in f0.files if k.startswith("group ")]
        for kind in ["frozen", "lora"]:
            f = np.load(files[(ds, kind)])
            runs = sorted({k.rsplit("|", 2)[0] for k in f.files if "|" in k})  # "model|seed"
            for run in runs:
                for o in PRIMARY:
                    for m in ["nll", "brier"]:
                        a, b = f[f"{run}|grsats|{m}"].astype(float), f[f"{run}|{o}|{m}"].astype(float)
                        wg = lambda s: np.max([(counts[:, g] @ s[g]) / counts[:, g].sum(1) for g in masks], axis=0)
                        diffs[(o, m)].append(wg(a) - wg(b))
                        point[(o, m)].append(max(a[g].mean() for g in masks) - max(b[g].mean() for g in masks))
    print("\nsuccess test: grsats minus each primary comparison, averaged over 8 settings x 3 seeds")
    out = {}
    for (o, m), d in diffs.items():
        d = np.mean(d, axis=0)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[(o, m)] = (np.mean(point[(o, m)]), lo, hi)
        print(f"  wg {m:5s} grsats - {o:8s} {out[(o, m)][0]:+.4f}  95% ci [{lo:+.4f}, {hi:+.4f}]")
    return out


def main():
    res = load()
    friedman(res)
    h2(res)
    pooled_bootstrap()


if __name__ == "__main__":
    main()

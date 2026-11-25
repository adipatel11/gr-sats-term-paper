# fewer and bigger figures for the simple version of the poster. same csvs and drawing functions as
# make_poster_figures.py. saves to Phase 3/Poster/Simple/figures
# the corruption strip (2_corruptions.png) is copied from Phase 3/Poster/figures, make_poster_figures.py draws it
# python make_simple_poster_figures.py
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import make_poster_figures as mp
import make_tables as mt
import rank_tests
from config import ROOT

OLD = mp.OUT
mp.OUT = ROOT / "Phase 3" / "Poster" / "Simple" / "figures"
W = mp.W
MAROON = "#5d1725"  # official mississippi state maroon, same as the bars and title band of this poster
mp.TEAL = MAROON  # table headers are drawn in make_poster_figures.py
mt.COLORS["grsats"] = MAROON  # only on this poster, the paper keeps its own color
plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 15.5,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})


def worst(d):
    # worst-group nll by method and severity: worst group of every run, then the mean over the runs
    d = d[d["group"] != "all"]
    v = d.groupby(["dataset", "model", "seed", "method", "severity", "group"])["nll"].mean()
    v = v.groupby(["dataset", "model", "seed", "method", "severity"]).max()
    return v.groupby(["method", "severity"]).mean().unstack()


def in_pool(det):
    ok = det["family"].eq("clean")
    for ds, fams in mt.POOL_FAMS.items():
        ok |= det["dataset"].eq(ds) & det["family"].isin(fams)
    return det[ok & det["method"].isin(mt.CURVES)]


def row(name, panels):
    # one row of four panels with a legend on top. panels are (title, ylabel, curves, severities)
    fig, axes = plt.subplots(1, 4, figsize=(W, 3.1))
    for ax, (title, ylabel, curves, sevs, yt) in zip(axes, panels):
        mp.lines(ax, curves, sevs)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_yticks(yt)
        ax.set_xlabel("Severity (0 = clean)" if sevs[0] == 0 else "Severity")
        ax.xaxis.label.set_size(14)
    for pair in (axes[:2], axes[2:4]):  # same y range inside a figure, except figure 6 (two very different groups)
        if pair[0].get_title() not in ["60+ (worst)"]:
            lo = min(a.get_ylim()[0] for a in pair)
            hi = max(a.get_ylim()[1] for a in pair)
            for a in pair:
                a.set_ylim(lo, hi)
    fig.tight_layout(rect=(0, 0, 1, 0.87), pad=0.4, w_pad=1.0)
    mp.top_legend(fig, axes[0], 6, columnspacing=1.2, handlelength=1.7)
    mp.save(fig, name)


def results(det):
    d = in_pool(det)
    all_sev, corr = list(range(6)), list(range(1, 6))
    lp = d["model"].str.endswith(" lp")
    t = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    row("4_severity_adaptation.png",
        [("HAM10000", "WG NLL", worst(d[d["dataset"] == "ham"]), all_sev, t),
         ("Fitzpatrick17k", "WG NLL", worst(d[d["dataset"] == "fitz"]), all_sev, t),
         ("LP", "WG NLL", worst(d[lp]), all_sev, t), ("LoRA", "WG NLL", worst(d[~lp]), all_sev, t)])
    noise, other = d[d["family"] == "noise"], d[~d["family"].isin(["noise", "clean"])]
    row("5_noise_tradeoff.png",
        [("Noise", "WG NLL", worst(noise), corr, [0.4, 0.6, 0.8]),
         ("Other families", "WG NLL", worst(other), corr, [0.4, 0.6, 0.8]),
         ("60+ (worst)", "NLL", mt.sev_curves(det, "ham", "nll", "60+"), all_sev, [0.3, 0.4, 0.5, 0.6]),
         ("<60 (best)", "NLL", mt.sev_curves(det, "ham", "nll", "<60"), all_sev, [0.10, 0.14, 0.18])])
    # numbers for the text
    for name, x in [("lp", d[lp]), ("lora", d[~lp]), ("other", other), ("noise", noise)]:
        x = x[x["severity"] >= 3]
        v = x[x["group"] != "all"].groupby(["dataset", "model", "seed", "method", "group"])["nll"].mean()
        v = v.groupby(["dataset", "model", "seed", "method"]).max().groupby("method").mean()
        print(name, v.round(4).to_dict(), "gain over ts_corr", round(v["ts_corr"] - v["grsats"], 4))


def framework():
    # figure 2 of the paper with fewer words
    H = 4.1
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, text, mine=False, size=18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#f1e3e6" if mine else "white", ec=MAROON if mine else "#777777",
                                    lw=2.5 if mine else 1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.3)

    def arrow(*pts):
        ax.plot(*zip(*pts), color="#444444", lw=2)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=2, shrinkA=0, shrinkB=0, mutation_scale=20))

    box(0.08, 2.2, 1.15, 1.1, "Image\n$x$")
    box(1.6, 2.2, 1.65, 1.1, "Adapted\nViT-B")
    box(3.85, 3.1, 1.9, 0.9, "Logit $z(x)$")
    box(3.85, 1.55, 1.9, 0.95, "Distance to\ntraining set")
    box(6.2, 1.55, 1.75, 0.95, "Shift score\n$s(x)$", True)
    box(8.4, 1.55, 2.62, 2.45, "Calibrator\n$\\beta=a_0+a_1 s(x)$\n$q(x)=\\sigma(\\beta\\,z(x))$", True, 19)
    box(5.3, 0.14, 2.65, 0.95, "Calibration images\nwith group labels", size=17)
    box(8.4, 0.14, 2.62, 0.95, "Worst-group fit\n$\\min_{\\theta}\\,\\max_{g}\\,L_g(\\theta)$", True)
    ax.add_patch(Rectangle((5.18, 0.04), 5.88, 1.15, fc="none", ec="#777777", lw=1.4, ls="--"))
    ax.text(5.1, 0.82, "Only while fitting", ha="right", va="center", style="italic", color="#555555", fontsize=17)
    ax.text(5.1, 0.4, "Shaded = my method", ha="right", va="center", color=MAROON, fontweight="bold", fontsize=17)
    arrow((1.23, 2.75), (1.6, 2.75))
    arrow((3.25, 2.75), (3.5, 2.75), (3.5, 3.55), (3.85, 3.55))
    arrow((3.25, 2.75), (3.5, 2.75), (3.5, 2.02), (3.85, 2.02))
    arrow((5.75, 3.55), (8.4, 3.55))
    arrow((5.75, 2.02), (6.2, 2.02))
    arrow((7.95, 2.02), (8.4, 2.02))
    arrow((7.95, 0.62), (8.4, 0.62))
    arrow((9.71, 1.09), (9.71, 1.55))
    mp.save(fig, "2_framework.png")


def equation():
    fig = plt.figure(figsize=(5.2, 1.0))
    fig.text(0.5, 0.5, r"$L_g(\theta)=\frac{1}{6}\sum_{\ell=0}^{5}\ \frac{1}{|D_{g,\ell}|}\sum_{D_{g,\ell}}-\log q_\theta(y\,|\,x)$",
             ha="center", va="center", fontsize=21)
    mp.save(fig, "3_equation.png")


def tables(set_mean, ranks):
    mp.table("T1_data.png", 6.0, [0.05, 4.2, 5.95], [
        ["Images (malignant)", "HAM10000", "Fitz17k-C"],
        ["Training", "8,164 (1,740)", "7,779 (1,144)"],
        ["Calibration and test", "1,784 (212)", "3,319 (479)"],
        ["Female", "820 (80)", ""], ["Male", "964 (132)", ""],
        ["Under 60", "1,206 (62)", ""], ["60 and over", "578 (150)", ""],
        ["Skin type I-II", "", "1,794 (287)"], ["Skin type III-IV", "", "1,163 (160)"],
        ["Skin type V-VI", "", "362 (32)"],
    ], ["left", "right", "right"], rule_after=(3,), size=16, rowh=0.36)

    mp.table("T2_baselines.png", W, [0.1, 4.3, 9.0], [
        ["Method", "Temperature", "Fit for"],
        ["TS (clean, noise, corruptions)", "Same for every image", "Average"],
        ["DAC [1]", "From k-NN distance", "Average"],
        ["AAR [2]", "From how atypical the image is", "Average"],
        ["Multi-domain TS", "Predicted from image features", "Average"],
        ["Cluster-wise TS", "One per image cluster", "Each cluster"],
        ["GR-SATS", "From the shift score", "Worst group"],
    ], ["left"] * 3, bold={(6, 0), (6, 1), (6, 2)}, size=17, rowh=0.37)

    avg = set_mean.groupby("method").mean()
    avg["rank"] = ranks
    cols = ["wg_nll", "wg_brier", "clean_ece", "rank"]
    best = {c: avg.loc[rank_tests.MAIN, c].min() for c in cols}
    rows, bold = [["Method", "WG NLL", "WG Brier", "Clean ECE", "Rank"]], set()
    for m in rank_tests.MAIN:
        cells = []
        for k, c in enumerate(cols):
            d = 2 if c == "rank" else 3
            cells.append(f"{avg.loc[m, c]:.{d}f}")
            if cells[-1] == f"{best[c]:.{d}f}":
                bold.add((len(rows), k + 1))
        if m == "grsats":
            bold.add((len(rows), 0))
        rows.append([mt.short_name(m)] + cells)
    mp.table("T3_results.png", 6.6, [0.05, 3.2, 4.4, 5.75, 6.5], rows, ["left"] + ["right"] * 4, bold=bold, size=15,
             rowh=0.35)


def main():
    res, det = mt.load("extra"), mt.load("detail")
    set_mean = mt.per_setting(res)
    ranks = rank_tests.friedman(rank_tests.load())[0]
    mp.groups(det)
    framework()
    equation()
    results(det)
    tables(set_mean, ranks)
    shutil.copy(OLD / "2_corruptions.png", mp.OUT / "1_corruptions.png")
    (mp.OUT / "1_groups.png").replace(mp.OUT / "0_groups.png")


if __name__ == "__main__":
    main()

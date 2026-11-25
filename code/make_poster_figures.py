# poster versions of the paper's figures and tables, drawn at their real size on the 36 x 24 inch poster
# so the text is big enough to read. same csvs and same functions as make_tables.py
# saves to Phase 3/Poster/figures
# python make_poster_figures.py
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

import make_tables as mt
import rank_tests
from config import FAMILIES, ROOT

OUT = ROOT / "Phase 3" / "Poster" / "figures"
TEAL = "#264e5b"  # section bars of the poster
W = 11.1  # full width inside a column

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "cm", "font.size": 14, "axes.titlesize": 14,
                     "axes.labelsize": 14, "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300)  # no tight crop, so the size in inches stays exact
    plt.close(fig)
    print(f"wrote {name}")


def lines(ax, curves, sevs):
    for m in mt.CURVES:
        ax.plot(sevs, curves.loc[m, sevs].to_numpy(), color=mt.COLORS[m], marker="o", ms=5,
                lw=3 if m == "grsats" else 1.8, ls="--" if m in ["none", "ts_clean"] else "-", label=mt.short_name(m))
    ax.set_xticks(sevs)
    ax.grid(alpha=0.3)


def top_legend(fig, ax, ncol, **kw):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=ncol, frameon=False,
               borderaxespad=0.1, **kw)


def severity(set_mean):
    fig, axes = plt.subplots(2, 4, figsize=(W, 4.3), sharex=True)
    for ax, (ds, model) in zip(axes.ravel(), mt.ORDER):
        c = set_mean.loc[(ds, model), [f"wg_nll sev{l}" for l in range(6)]]
        c.columns = range(6)
        lines(ax, c, list(range(6)))
        bb, kind = model.split()
        ax.set_title(f"{mt.SHORT[ds]}, {mt.SHORT[kind]}, {mt.SHORT[bb]}", fontsize=13)
    for ax in axes.ravel():
        ax.set_ylabel("WG NLL")
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = clean)")
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=1.2, h_pad=0.4)
    top_legend(fig, axes[0, 0], 6, columnspacing=1.4, handlelength=1.8)
    save(fig, "4_severity.png")


def two_panels(name, panels, titles, ylabel, ticks):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55))
    for ax, c, t, yt in zip(axes, panels, titles, ticks):
        lines(ax, c, list(range(6)))
        ax.set_yticks(yt)
        ax.set_title(t)
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_ylabel(ylabel)
    fig.tight_layout(rect=(0, 0, 1, 0.82), pad=0.4, w_pad=1.2)
    top_legend(fig, axes[0], 3, fontsize=12.5, columnspacing=1.2, handlelength=1.8, labelspacing=0.2)
    save(fig, name)


def families(det):
    fams = list(FAMILIES)
    fig, axes = plt.subplots(2, len(fams), figsize=(W, 3.2), sharex=True, sharey="row")
    for row, ds in zip(axes, ["ham", "fitz"]):
        d = det[(det["dataset"] == ds) & det["method"].isin(mt.CURVES) & (det["group"] != "all")]
        for ax, fam in zip(row, fams):
            v = d[d["family"] == fam].groupby(["model", "seed", "method", "severity", "group"])["nll"].mean()
            c = v.groupby(["model", "seed", "method", "severity"]).max().groupby(["method", "severity"]).mean()
            lines(ax, c.unstack(), list(range(1, 6)))
            ax.set_title(f"{mt.SHORT[ds]}, {fam}" + ("*" if fam not in mt.POOL_FAMS[ds] else ""), fontsize=13)
        for ax in row:
            ax.set_ylabel("WG NLL")
            ax.tick_params(labelleft=True)
            ax.set_yticks([0.4, 0.6, 0.8, 1.0] if ds == "ham" else [0.4, 0.6])
    for ax in axes[1]:
        ax.set_xlabel("Severity")
    fig.tight_layout(rect=(0, 0, 1, 0.91), pad=0.4, w_pad=1.2, h_pad=0.3)
    top_legend(fig, axes[0, 0], 6, columnspacing=1.4, handlelength=1.8)
    save(fig, "7_families.png")


def groups(det):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for ax, ds in zip(axes, ["ham", "fitz"]):
        d = det[(det["dataset"] == ds) & (det["method"] == "ts_clean") & (det["group"] != "all")
                & det["family"].isin(mt.POOL_FAMS[ds] + ["clean"])]
        curve = d.groupby(["model", "seed", "group", "severity"])["nll"].mean().groupby(["group", "severity"]).mean()
        for g in curve.index.get_level_values(0).unique():
            ax.plot(range(6), curve.loc[g].to_numpy(), marker="o", ms=4, lw=1.8, label=g)
        ax.set_title(mt.LONG[ds])
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_xticks(range(6))
        ax.set_ylim(0, curve.max() * 1.6)
        ax.grid(alpha=0.3)
        pairs = sorted(zip(*ax.get_legend_handles_labels()), key=lambda p: -p[0].get_ydata()[-1])
        ax.legend(*zip(*pairs), loc="upper left", ncol=2, frameon=False, handlelength=1, columnspacing=0.6,
                  fontsize=12, labelspacing=0.2, borderaxespad=0.1)
        ax.set_ylabel("NLL, clean-fit TS")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    save(fig, "1_groups.png")


def corruption_strip():
    # one ham10000 test image and 11 of the 15 corruptions at severity 5, made with the same medmnist-c
    # settings as the experiments
    from data import load_ham
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    cor = CORRUPTIONS_DS["dermamnist"]
    df = load_ham()
    row = df[(df["split"] == "test") & (df["y"] == 1)].iloc[3]
    img = Image.open(row["path"]).convert("RGB")
    s = 224 / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    x, y = (img.width - 224) // 2, (img.height - 224) // 2
    img = img.crop((x, y, x + 224, y + 224))
    tiles = [("Clean", [("clean", None)]), ("Digital", [("pixelate", "pixelate"), ("JPEG", "jpeg_compression")]),
             ("Noise", [("Gaussian", "gaussian_noise"), ("impulse", "impulse_noise")]),
             ("Blur", [("defocus", "defocus_blur"), ("zoom", "zoom_blur")]),
             ("Color", [("brightness", "brightness_up"), ("contrast", "contrast_down")]),
             ("Task-specific", [("corners", "black_corner"), ("text", "characters")])]
    colors = ["#444444", "#8c1515", "#b36b00", "#3b7a8c", "#6a3d9a", "#1f4fbf"]
    fig = plt.figure(figsize=(W, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 1.75)
    ax.axis("off")
    size, gap = 0.96, 0.1
    x = (W - 11 * size - 5 * gap) / 2
    for (fam, items), color in zip(tiles, colors):
        start = x
        for label, key in items:
            a = img
            if key:
                random.seed(0)
                np.random.seed(0)
                cor[key].rng = np.random.default_rng(0)
                a = Image.fromarray(np.ascontiguousarray(cor[key].apply(img, 4)))
            ax.imshow(np.asarray(a), extent=(x, x + size, 0.78, 0.78 + size), interpolation="lanczos")
            if key:
                ax.text(x + size / 2, 0.74, label, ha="center", va="top", fontsize=13, color=color)
            x += size
        ax.plot([start + 0.04, start + 0.04, x - 0.04, x - 0.04], [0.48, 0.40, 0.40, 0.48] if len(items) > 1
                else [0.40] * 4, color=color, lw=2)
        ax.text((start + x) / 2, 0.33, fam, ha="center", va="top", fontsize=16, color=color, fontweight="bold")
        x += gap
    ax.set_xlim(0, W)
    ax.set_ylim(0, 1.75)
    ax.set_aspect("auto")
    print("image used in the strip:", row["id"])
    save(fig, "2_corruptions.png")


def framework():
    # figure 2 of the paper, redrawn with bigger text
    H = 4.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, text, mine=False, size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#dbe8ec" if mine else "white", ec=TEAL if mine else "#777777",
                                    lw=2.5 if mine else 1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.35)

    def arrow(*pts):
        ax.plot(*zip(*pts[:-1], pts[-1]), color="#444444", lw=1.8, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=18))

    box(0.08, 2.1, 1.6, 1.15, "Image $x$,\nclean or\ncorrupted")
    box(2.0, 2.1, 1.8, 1.15, "Adapted ViT-B\nDINOv2 or\nBiomedCLIP,\nLP or LoRA", size=13)
    box(4.35, 3.1, 1.75, 0.85, "Linear head\ngives logit $z(x)$", size=13)
    box(4.35, 1.5, 1.75, 1.15, "Distance $d_k(x)$\nto $k$-th nearest\ntraining image", size=13)
    box(6.4, 1.5, 1.85, 1.15, "C1: shift score\n$s(x)$, log scale,\nclipped", True, 13)
    box(8.6, 1.5, 2.42, 2.45, "C2: calibrator\n$\\beta(s)=a_0+a_1 s$\n$q(x)=\\sigma(\\beta(s)\\,z(x))$", True, 15)
    box(6.4, 0.14, 1.85, 0.95, "Fitting folds\nwith group labels", size=13)
    box(8.6, 0.14, 2.42, 0.95, "C3: worst-group fit\n$\\min_{\\theta}\\,\\max_{g}\\,L_g(\\theta)$", True, 14)
    ax.add_patch(Rectangle((6.28, 0.04), 4.78, 1.15, fc="none", ec="#777777", lw=1.3, ls="--"))
    ax.text(6.2, 0.62, "Used only while fitting", ha="right", va="center", style="italic", color="#555555", fontsize=14)
    ax.text(0.08, 0.62, "Shaded = proposed", ha="left", va="center", color=TEAL, fontweight="bold", fontsize=14)
    arrow((1.68, 2.68), (2.0, 2.68))
    arrow((3.8, 2.68), (4.07, 2.68), (4.07, 3.52), (4.35, 3.52))
    arrow((3.8, 2.68), (4.07, 2.68), (4.07, 2.07), (4.35, 2.07))
    arrow((6.1, 3.52), (8.6, 3.52))
    arrow((6.1, 2.07), (6.4, 2.07))
    arrow((8.25, 2.07), (8.6, 2.07))
    arrow((8.25, 0.62), (8.6, 0.62))
    arrow((9.8, 1.09), (9.8, 1.5))
    ax.text(9.9, 1.33, "$\\hat{\\theta}$", ha="left", va="center", fontsize=15)
    save(fig, "3_framework.png")


def equations():
    fig = plt.figure(figsize=(W, 0.85))
    eqs = [r"$s(x)=\mathrm{clip}\left(\frac{\log d_k(x)-m}{r},\ s_{\min},\ s_{\max}\right)$",
           r"$L_g(\theta)=\frac{1}{6}\sum_{\ell=0}^{5}\frac{1}{|D_{g,\ell}|}\sum_{(x,y)\in D_{g,\ell}}-\log q_\theta(y\,|\,x)$"]
    for x, e in zip([0.25, 0.73], eqs):
        fig.text(x, 0.5, e, ha="center", va="center", fontsize=19)
    save(fig, "3_equations.png")


def table(name, width, colx, rows, aligns, rule_after=(), bold=(), size=14, rowh=0.3, heads=1):
    # plain table with a rule on top, under the header and at the bottom. bold has (row, col) pairs
    H = rowh * len(rows) + 0.12
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 0.05
    for r, row in enumerate(rows):
        y = top - rowh * (r + 0.5)
        for c, (text, x) in enumerate(zip(row, colx)):
            ax.text(x, y, text, ha=aligns[c], va="center_baseline", fontsize=size,
                    fontweight="bold" if r < heads or (r, c) in bold else "normal",
                    color=TEAL if r < heads else "black")
    for k, lw in [(0, 2), (heads, 1.2), (len(rows), 2)] + [(r, 0.8) for r in rule_after]:
        ax.plot([0.02, width - 0.02], [top - rowh * k] * 2, color="black", lw=lw)
    save(fig, name)


def tables(set_mean, ranks, boot):
    table("T1_literature.png", W, [0.05, 2.75, 7.2], [
        ["Study", "What it does", "Limitation"],
        ["DAC [1], ICML 2023", "Per-image temperature from k-NN distances", "Average only, no groups or medical data"],
        ["AAR [2], NeurIPS 2023", "Logit scale from image atypicality", "Average loss, clean data only"],
        ["Cluster-Focal [3], MICCAI 2023", "Training loss for worst-group calibration", "Needs retraining, clean test data"],
        ["FairMedFM [4], NeurIPS 2024", "Fairness test of 20 foundation models", "Clean tests, no calibration method"],
        ["MedMNIST-C [5], 2024", "Image corruptions at five severities", "No calibration metrics, no groups"],
        ["Roschewitz [6], TMLR 2025", "Compares calibrators under natural shift", "No subgroups, no graded corruption"],
    ], ["left"] * 3, size=12.5, rowh=0.275)

    table("T2_data.png", 5.6, [0.05, 3.75, 5.5], [
        ["Images (malignant)", "HAM10000", "Fitzpatrick17k-C"],
        ["Training split", "8,164 (1,740)", "7,779 (1,144)"],
        ["Pool (val + test)", "1,784 (212)", "3,319 (479)"],
        ["Female | Male", "820 (80) | 964 (132)", ""],
        ["Under 60 | 60+", "1,206 (62) | 578 (150)", ""],
        ["Skin type I-II", "", "1,794 (287)"],
        ["Skin type III-IV", "", "1,163 (160)"],
        ["Skin type V-VI", "", "362 (32)"],
    ], ["left", "right", "right"], rule_after=(3,), size=12.5, rowh=0.275)

    avg = set_mean.groupby("method").mean()
    avg["rank"] = ranks
    cols = ["wg_nll", "wg_nll no noise", "wg_brier", "clean_ece", "rank"]
    rows = [["Method", "WG NLL", "No noise", "WG Brier", "Clean ECE", "Rank"]]
    best = {c: min(avg.loc[rank_tests.MAIN, c]) for c in cols}
    bold = set()
    for m in rank_tests.MAIN + ["group_grsats"]:
        oracle = m == "group_grsats"
        cells = []
        for k, c in enumerate(cols):
            d = 2 if c == "rank" else 3
            cells.append("n/a" if oracle and c == "rank" else f"{avg.loc[m, c]:.{d}f}")
            if not oracle and cells[-1] == f"{best[c]:.{d}f}":
                bold.add((len(rows), k + 1))
        if m == "grsats":
            bold.add((len(rows), 0))
        rows.append([mt.short_name(m).replace("$^\\dagger$", " (oracle)")] + cells)
    table("T3_results.png", 6.5, [0.05, 3.15, 4.0, 4.9, 5.85, 6.45], rows, ["left"] + ["right"] * 5,
          rule_after=(10,), bold=bold, size=12.5, rowh=0.26)

    rows = [["GR-SATS minus", "WG NLL [95% CI]", "WG Brier [95% CI]", "Wins/ties/losses"]]
    wins = {"ts_corr": "24 / 0 / 0", "dac": "22 / 1 / 1", "aar": "18 / 5 / 1"}
    for o in rank_tests.PRIMARY:
        a, b = boot[(o, "nll")], boot[(o, "brier")]
        rows.append([mt.short_name(o), f"{a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]",
                     f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]", wins[o]])
    rows = [[c.replace("-0", "−0") for c in r] for r in rows]
    table("T4_success.png", W, [0.05, 5.3, 8.9, 11.0], rows, ["left"] + ["right"] * 3, size=14, rowh=0.3)


def main():
    res, det = mt.load("extra"), mt.load("detail")
    set_mean = mt.per_setting(res)
    ranks = rank_tests.friedman(rank_tests.load())[0]
    boot = rank_tests.pooled_bootstrap()
    groups(det)
    framework()
    equations()
    severity(set_mean)
    two_panels("5_brier.png", [mt.sev_curves(det, ds, "brier") for ds in ["ham", "fitz"]],
               ["HAM10000", "Fitzpatrick17k"], "WG Brier", [[0.10, 0.13, 0.16], [0.08, 0.10]])
    two_panels("6_tradeoff.png", [mt.sev_curves(det, "ham", "nll", g) for g in ["60+", "<60"]],
               ["HAM, 60+ (worst)", "HAM, <60 (best)"], "NLL", [[0.3, 0.4, 0.5, 0.6], [0.10, 0.14, 0.18]])
    families(det)
    tables(set_mean, ranks, boot)
    corruption_strip()


if __name__ == "__main__":
    main()

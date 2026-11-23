# makes the paper's tables (latex, in tables/) and figures (matplotlib pdfs, in figures/) from the
# csvs in results/. every number in the paper comes from here or from rank_tests.py
# python make_tables.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rank_tests
from config import FAMILIES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"

NAMES = {
    "none": "No calibration", "ts_clean": "TS, clean", "ts_gauss": "TS, Gaussian noise",
    "ts_corr": "TS, corruptions", "dac": "DAC", "md_ts": "Multi-domain TS", "aar": "AAR",
    "cluster_ts": "Cluster-wise TS", "grsats": "GR-SATS (proposed)",
    "group_ts": "Group-wise TS$^\\dagger$", "group_grsats": "Per-group GR-SATS$^\\dagger$",
}
ORACLES = ["group_ts", "group_grsats"]
# one color per method, the same in every figure (okabe-ito, safe for colorblind readers)
COLORS = {"none": "#56b4e9", "ts_clean": "#e69f00", "ts_corr": "#009e73", "dac": "#cc79a7", "aar": "#0072b2",
          "grsats": "#d55e00"}
# order of the 8 settings in the tables and the severity figure
ORDER = [(ds, f"{bb} {kind}") for ds in ["ham", "fitz"] for kind in ["lp", "lora"] for bb in ["dinov2", "biomedclip"]]
SHORT = {"ham": "HAM", "fitz": "Fitz", "dinov2": "DINOv2", "biomedclip": "BiomedCLIP", "lp": "LP", "lora": "LoRA"}
# which corruptions are in each dataset's main test pool (fitz leaves out the task-specific family)
POOL_FAMS = {"ham": list(FAMILIES), "fitz": [f for f in FAMILIES if f != "task-specific"]}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(kind):
    return pd.concat([pd.read_csv(RESULTS / f"{ds}_{k}_{kind}.csv").assign(dataset=ds)
                      for ds, k in rank_tests.SETTINGS])


def per_setting(res):
    # mean over the 3 seeds, one row per (dataset, model, method)
    return res.groupby(["dataset", "model", "method"]).mean(numeric_only=True)


def fmt(v, best, digits=3):
    # bold when it rounds to the best value, so ties in the printed table are both bold.
    # best is None for rows that can't be bold (the oracles)
    s = f"{v:.{digits}f}"
    return f"\\textbf{{{s}}}" if best is not None and s == f"{best:.{digits}f}" else s


def write(name, lines):
    TABLES.mkdir(exist_ok=True)
    (TABLES / name).write_text("\n".join(lines) + "\n")
    print(f"wrote tables/{name}")


def main_table(set_mean, ranks):
    # averaged over the 8 settings. best non-oracle value in each column is bold
    avg = set_mean.groupby("method").mean()
    avg["rank"] = ranks
    cols = [("wg_nll", "WG NLL", min), ("wg_nll no noise", "WG NLL, no noise", min), ("wg_brier", "WG Brier", min),
            ("nll", "NLL", min), ("clean_ece", "Clean ECE", min), ("clean_auc", "Clean AUC", max), ("rank", "Rank", min)]
    methods = rank_tests.MAIN
    best = {c: f(avg.loc[methods, c]) for c, _, f in cols}
    lines = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
             "\\textbf{Method} & " + " & ".join(f"\\textbf{{{h}}}" for _, h, _ in cols) + " \\\\", "\\midrule"]
    for m in methods + ORACLES:
        if m == ORACLES[0]:
            lines.append("\\midrule")
        cells = []
        for c, _, _ in cols:
            v = avg.loc[m, c]
            if c == "rank":
                cells.append("n/a" if m in ORACLES else fmt(v, best[c], 2))
            else:
                cells.append(fmt(v, None if m in ORACLES else best[c]))
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    write("results.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def setting_table(set_mean):
    # worst-group nll for each of the 8 settings
    wg = set_mean["wg_nll"].unstack("method")
    lines = ["\\begin{tabular}{@{}l" + "c" * 8 + "@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{\\textbf{HAM10000}} & \\multicolumn{4}{c}{\\textbf{Fitzpatrick17k}} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
             " & \\multicolumn{2}{c}{LP} & \\multicolumn{2}{c}{LoRA} & \\multicolumn{2}{c}{LP} & \\multicolumn{2}{c}{LoRA} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
             "\\textbf{Method} & " + " & ".join(SHORT[m.split()[0]] for _, m in ORDER) + " \\\\", "\\midrule"]
    best = {s: wg.loc[s, rank_tests.MAIN].min() for s in ORDER}
    for m in rank_tests.MAIN + ORACLES:
        if m == ORACLES[0]:
            lines.append("\\midrule")
        lines.append(f"{NAMES[m]} & " + " & ".join(fmt(wg.loc[s, m], None if m in ORACLES else best[s])
                                                    for s in ORDER) + " \\\\")
    write("settings.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def ablation_table(set_mean):
    rows = [("Global", "Average (TS with corruptions)", "ts_corr"), ("Global", "Worst group", "ts_wg"),
            ("Shift-aware", "Average", "sats_avg"), ("Shift-aware", "Worst group (GR-SATS)", "grsats"), None,
            ("GR-SATS with excess NLL", None, "grsats_excess"), ("GR-SATS with bias term", None, "grsats_bias"),
            ("DAC, average squared error", None, "dac"), ("DAC, average NLL", None, "dac_nll"),
            ("DAC-WG, worst-group NLL", None, "dac_wg"), None,
            ("GR-SATS, $k = 1$", None, "grsats_k1"), ("GR-SATS, $k = 50$", None, "grsats_k50"),
            ("GR-SATS, group-balanced bank", None, "grsats_bal")]
    avg = set_mean.groupby("method").mean()
    by_ds = set_mean.groupby(["dataset", "method"]).mean()
    cols = {"WG NLL": avg["wg_nll"], "HAM": by_ds.loc["ham", "wg_nll"], "Fitz": by_ds.loc["fitz", "wg_nll"],
            "NLL": avg["nll"]}
    used = [r[2] for r in rows if r]
    best = {h: v[used].min() for h, v in cols.items()}
    lines = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
             "\\textbf{Temperature} & \\textbf{Loss} & " + " & ".join(f"\\textbf{{{h}}}" for h in cols) + " \\\\",
             "\\midrule"]
    for r in rows:
        if r is None:
            lines.append("\\midrule")
            continue
        a, b, m = r
        label = f"{a} & {b}" if b else f"\\multicolumn{{2}}{{@{{}}l}}{{{a}}}"
        lines.append(label + " & " + " & ".join(fmt(v[m], best[h]) for h, v in cols.items()) + " \\\\")
    write("ablation.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def success_table(boot):
    lines = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule",
             "\\textbf{GR-SATS minus} & \\textbf{WG NLL [95\\% CI]} & \\textbf{WG Brier [95\\% CI]} \\\\", "\\midrule"]
    for o in rank_tests.PRIMARY:
        cells = []
        for m, d in [("nll", 3), ("brier", 4)]:
            p, lo, hi = boot[(o, m)]
            cells.append(f"${p:+.{d}f}$ [${lo:+.{d}f}$, ${hi:+.{d}f}$]")
        lines.append(f"{NAMES[o]} & " + " & ".join(cells) + " \\\\")
    write("success.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def worst_group(det):
    # worst group = highest mean nll over the rows given (same number of images in every row of a group)
    return det[det["group"] != "all"].groupby(["dataset", "model", "seed", "method", "group"])["nll"].mean() \
        .groupby(["dataset", "model", "seed", "method"]).max()


def family_table(det):
    # worst-group nll at severities 3 to 5 when each family is the held-out one, mean over the 4
    # settings of a dataset and 3 seeds
    methods = ["ts_corr", "dac", "aar", "grsats"]
    det = det[det["method"].isin(methods) & (det["severity"] >= 3)]
    vals = {fam: worst_group(det[det["family"] == fam]).groupby(["dataset", "method"]).mean() for fam in FAMILIES}
    head = " & ".join(["TS-C", "DAC", "AAR", "GR"] * 2)
    lines = ["\\begin{tabular}{@{}l" + "c" * 8 + "@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{\\textbf{HAM10000}} & \\multicolumn{4}{c}{\\textbf{Fitzpatrick17k}} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}", f"\\textbf{{Held-out family}} & {head} \\\\", "\\midrule"]
    for fam in FAMILIES:
        cells = []
        for ds in ["ham", "fitz"]:
            v = vals[fam].loc[ds]
            cells += [fmt(v[m], v.min()) for m in methods]
        name = fam.capitalize() + ("$^*$" if fam == "task-specific" else "")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    write("families.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


def severity_figure(set_mean):
    # worst-group nll against severity for every setting (h1 and h2)
    methods = ["none", "ts_clean", "ts_corr", "dac", "aar", "grsats"]
    fig, axes = plt.subplots(2, 4, figsize=(7, 3.1), sharex=True)
    for ax, (ds, model) in zip(axes.ravel(), ORDER):
        for m in methods:
            y = set_mean.loc[(ds, model, m), [f"wg_nll sev{l}" for l in range(6)]]
            ax.plot(range(6), y.to_numpy(), color=COLORS[m], marker="o", ms=2.5, lw=1.2 if m == "grsats" else 0.9,
                    ls="--" if m in ["none", "ts_clean"] else "-", label=NAMES[m].replace(" (proposed)", ""))
        bb, kind = model.split()
        ax.set_title(f"{SHORT[ds]}, {SHORT[kind]}, {SHORT[bb]}")
        ax.set_xticks(range(6))
        ax.grid(alpha=0.3, lw=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Worst-group NLL")
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = clean)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(methods), frameon=False)
    save(fig, "severity.pdf")


def groups_figure(det):
    # h1: per-group nll of clean-fit ts against severity, mean over the 4 settings of each dataset
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    for ax, ds in zip(axes, ["ham", "fitz"]):
        d = det[(det["dataset"] == ds) & (det["method"] == "ts_clean") & (det["group"] != "all")
                & det["family"].isin(POOL_FAMS[ds] + ["clean"])]
        curve = d.groupby(["model", "seed", "group", "severity"])["nll"].mean().groupby(["group", "severity"]).mean()
        for g in curve.index.get_level_values(0).unique():
            y = curve.loc[g].to_numpy()
            line, = ax.plot(range(6), y, marker="o", ms=2.5, lw=0.9)
            ax.text(5.25, y[-1], g, va="center", fontsize=6, color=line.get_color())  # label at the line's end
        ax.set_title("HAM10000" if ds == "ham" else "Fitzpatrick17k")
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_xticks(range(6))
        ax.set_xlim(-0.3, 6.4)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("NLL, clean-fit TS")
    fig.tight_layout()
    save(fig, "groups.pdf")


def reliability_figure(rel, set_mean):
    # reliability diagram of each dataset's worst group at severities 3 to 5. bins are equal-mass, so
    # bin b is averaged (weighted by images) over the 4 settings and 3 seeds
    methods = ["none", "ts_corr", "dac", "aar", "grsats"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85))
    for ax, ds in zip(axes, ["ham", "fitz"]):
        none = set_mean.loc[ds].xs("none", level="method")
        cols = [c for c in none.columns if c.startswith("nll ")]
        g = none[cols].mean().idxmax()[4:]  # group with the highest nll without calibration
        d = rel[(rel["dataset"] == ds) & (rel["condition"] == "sev3-5") & (rel["group"] == g)]
        for m in methods:
            x = d[d["method"] == m]
            w = x.groupby("bin")["n"].sum()
            risk = (x["mean risk"] * x["n"]).groupby(x["bin"]).sum() / w
            frac = (x["frac malignant"] * x["n"]).groupby(x["bin"]).sum() / w
            ax.plot(risk, frac, color=COLORS[m], marker="o", ms=2, lw=0.9, label=NAMES[m].replace(" (proposed)", ""))
        ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":")
        ax.set_title(f"{'HAM10000' if ds == 'ham' else 'Fitzpatrick17k'}, group {g}")
        ax.set_xlabel("Predicted risk")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Fraction malignant")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(methods), frameon=False,
               handlelength=1.2, columnspacing=0.8)
    save(fig, "reliability.pdf")


def cd_figure(ranks, cd):
    # critical difference diagram (demsar 2006): methods joined by a bar are not significantly different
    order = ranks.sort_values()
    k = len(order)
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.set_xlim(0.2, k + 0.8)
    ax.set_ylim(-1.15, 0.45)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.05], color="black", lw=0.8)
        ax.text(r, 0.1, str(r), ha="center", va="bottom")
    ax.plot([1, 1 + cd], [0.33, 0.33], color="black", lw=1.2)
    ax.text(1 + cd / 2, 0.36, f"CD = {cd:.2f}", ha="center", va="bottom")
    half = (k + 1) // 2
    for i, (m, r) in enumerate(order.items()):
        left = i < half
        y = -0.35 - 0.16 * (i if left else k - 1 - i)
        end = 0.6 if left else k + 0.4
        ax.plot([r, r, end], [0, y, y], color="black", lw=0.6)
        ax.text(end + (-0.05 if left else 0.05), y, f"{NAMES[m].replace(' (proposed)', '')} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=6)
    # bars over groups of methods whose ranks are within the critical difference
    cliques = []
    for i in range(k):
        j = max(j for j in range(i, k) if order.iloc[j] - order.iloc[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        y = -0.1 - 0.07 * n
        ax.plot([order.iloc[i] - 0.05, order.iloc[j] + 0.05], [y, y], color="black", lw=2)
    save(fig, "cd.pdf")


def main():
    res = load("extra")
    det = load("detail")
    rel = load("reliability")
    set_mean = per_setting(res)

    ranks, _, _, cd = rank_tests.friedman(rank_tests.load())
    boot = rank_tests.pooled_bootstrap()

    main_table(set_mean, ranks)
    setting_table(set_mean)
    ablation_table(set_mean)
    success_table(boot)
    family_table(det)
    severity_figure(set_mean)
    groups_figure(det)
    reliability_figure(rel, set_mean)
    cd_figure(ranks, cd)


if __name__ == "__main__":
    main()

"""Figures and tables for one experiment run.

  python faithfulness/plots.py --run_dir <dir>          (reads summary.json + records.jsonl)

Writes <run_dir>/figures/*.png and <run_dir>/tables/*.csv:
  fig01_baseline_confusion   gold label x predicted label
  fig02_removal_effects      flip rate and confidence drop: cited vs controls, by predicted label
  fig03_faithfulness_heatmap cited-minus-control for every contrast x predicted label (* = 95% CI excludes 0)
  fig04_transitions          how predictions move after removing cited vs random words
  fig05_drop_scatter         per example: confidence drop when removing cited vs random words
  fig06_consistency          predicted label x explanation stance (model) next to the human reference
  fig07_alignment            overlap of cited words with human-highlighted evidence
Colours: validated categorical slots (blue, orange, aqua, yellow), a one-hue blue ramp for
magnitudes, and a blue/grey/red diverging scale for signed differences.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

LABELS = ["entailment", "neutral", "contradiction"]
GROUPS = ["all"] + LABELS
SHORT = {"all": "All", "entailment": "Entailment", "neutral": "Neutral", "contradiction": "Contradiction", "none": "None"}

# palette (validated reference instance, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c43b3b", "#e88f8f", "#f0efec", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
})


def _save(fig, d, name):
    path = os.path.join(d, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _heat(ax, matrix, rows, cols, cmap, vmin, vmax, fmt, annot=None):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), [SHORT.get(c, c) for c in cols])
    ax.set_yticks(range(len(rows)), [SHORT.get(r, r) for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i][j]
            if v is None or v != v:
                ax.text(j, i, "n/a", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            t = annot[i][j] if annot else fmt.format(v)
            norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
            dark = norm > 0.6 if cmap is SEQ else abs(norm - 0.5) > 0.3
            ax.text(j, i, t, ha="center", va="center", fontsize=8.5, color="#ffffff" if dark else INK)
    return im


def _row_norm(conf, rows, cols):
    m = []
    for r in rows:
        tot = sum(conf[r].get(c, 0) for c in cols)
        m.append([conf[r].get(c, 0) / tot if tot else float("nan") for c in cols])
    return m


# ------------------------------------------------------------------ figures
def fig_confusion(s, d):
    conf = s["baseline"]["confusion_gold_x_pred"]
    m = _row_norm(conf, LABELS, LABELS)
    ann = [[f"{m[i][j]:.0%}\n({conf[r][c]})" for j, c in enumerate(LABELS)] for i, r in enumerate(LABELS)]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    _heat(ax, m, LABELS, LABELS, SEQ, 0, 1, "{:.0%}", ann)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    b = s["baseline"]
    ax.set_title(f"Baseline: accuracy {b['accuracy']:.1%}, macro-F1 {b['macro_f1']:.3f}")
    return _save(fig, d, "fig01_baseline_confusion")


def fig_removal(s, d, side="premise"):
    series = [(f"{side}:remove_cited", "Remove cited words"), (f"{side}:remove_random", "Remove random words"),
              (f"{side}:remove_overlap", "Remove other overlapping words"), (f"{side}:remove_gold", "Remove human-highlighted words")]
    series = [(k, n) for k, n in series if k in s["variants"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    w = 0.8 / len(series)
    for ax, metric, title in ((axes[0], "flip_rate", "Prediction changed (flip rate)"),
                              (axes[1], "prob_drop", "Confidence lost in original label")):
        for si, (k, name) in enumerate(series):
            xs, ys, lo, hi = [], [], [], []
            for gi, g in enumerate(GROUPS):
                x = s["variants"][k][g][metric]
                if x["mean"] is None:
                    continue
                xs.append(gi + (si - (len(series) - 1) / 2) * w)
                ys.append(x["mean"])
                lo.append(max(x["mean"] - x["lo"], 0))
                hi.append(max(x["hi"] - x["mean"], 0))
            ax.bar(xs, ys, width=w * 0.92, color=SLOT[si], label=name, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        ax.set_xticks(range(len(GROUPS)), [SHORT[g] for g in GROUPS])
        ax.set_xlabel("Model's predicted label")
        ax.set_title(title)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.18), ncol=len(series))
    fig.suptitle(f"Editing the {side}: does removing the cited evidence matter more than removing other words?",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    return _save(fig, d, f"fig02_removal_effects_{side}")


def fig_contrasts(s, d):
    names = [n for n, c in s["contrasts"].items() if c["all"]["flip_diff"]["n"] > 0]
    m, ann = [], []
    for n in names:
        row, arow = [], []
        for g in GROUPS:
            x = s["contrasts"][n][g]["flip_diff"]
            if x["mean"] is None:
                row.append(float("nan"))
                arow.append("n/a")
                continue
            sig = "*" if (x["lo"] > 0 or x["hi"] < 0) else ""
            row.append(x["mean"])
            arow.append(f"{x['mean']:+.2f}{sig}\nn={x['n']}")
        m.append(row)
        ann.append(arow)
    vmax = max([abs(v) for r in m for v in r if v == v] + [0.05])
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(names) + 1.6))
    im = _heat(ax, m, names, GROUPS, DIV, -vmax, vmax, "{:+.2f}", ann)
    ax.set_xlabel("Model's predicted label")
    ax.set_title("Faithfulness: cited minus control (flip-rate difference)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("> 0: cited evidence matters more (faithful)\n< 0: matters less than control", color=INK2, fontsize=8)
    fig.text(0.01, -0.02, "* 95% bootstrap CI excludes 0.  Isolation rows: keep-rate difference. "
             "Neutral predictions are not expected to flip when evidence is removed.", fontsize=8, color=MUTED)
    return _save(fig, d, "fig03_faithfulness_heatmap")


def fig_transitions(s, d):
    keys = [k for k in ("premise:remove_cited", "premise:remove_random", "hypothesis:remove_cited",
                        "hypothesis:remove_random") if k in s["transitions"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6))
    axes = [axes] if len(keys) == 1 else axes
    for ax, k in zip(axes, keys):
        conf = s["transitions"][k]
        m = _row_norm(conf, LABELS, LABELS)
        _heat(ax, m, LABELS, LABELS, SEQ, 0, 1, "{:.0%}")
        side, name = k.split(":")
        ax.set_title(f"{side}: {name.replace('_', ' ')}", fontsize=10)
        ax.set_xlabel("Prediction after edit")
    axes[0].set_ylabel("Original prediction")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    return _save(fig, d, "fig04_transitions")


def fig_scatter(records, d, side="premise"):
    a, b = f"{side}:remove_cited", f"{side}:remove_random"
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for li, l in enumerate(LABELS):
        xs, ys = [], []
        for r in records:
            if r["original"]["label"] != l or a not in r["variants"] or b not in r["variants"]:
                continue
            p0 = r["original"]["probs"][l]
            xs.append(p0 - r["variants"][b]["probs"][l])
            ys.append(p0 - r["variants"][a]["probs"][l])
        ax.scatter(xs, ys, s=14, alpha=0.55, color=SLOT[li], edgecolors=SURFACE, linewidths=0.4,
                   label=f"{SHORT[l]} (n={len(xs)})")
    ax.plot([-1, 1], [-1, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_xlabel("Confidence drop: remove random words")
    ax.set_ylabel("Confidence drop: remove cited words")
    ax.set_aspect("equal")
    ax.set_title(f"Per example ({side}): cited vs random removal")
    ax.text(-0.95, 0.9, "above diagonal:\ncited evidence matters more", fontsize=8, color=MUTED, va="top")
    ax.legend(title="Predicted label", loc="lower right", fontsize=8, title_fontsize=8)
    return _save(fig, d, f"fig05_drop_scatter_{side}")


def fig_consistency(s, d):
    c = s["consistency"]
    cols = LABELS
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    m1 = _row_norm(c["model_confusion_pred_x_stance"], LABELS, cols)
    _heat(axes[0], m1, LABELS, cols, SEQ, 0, 1, "{:.0%}")
    axes[0].set_title(f"Model: agreement {c['model']['all']['mean']:.1%}")
    axes[0].set_ylabel("Model's predicted label")
    m2 = _row_norm(c["human_confusion_gold_x_stance"], LABELS, cols)
    _heat(axes[1], m2, LABELS, cols, SEQ, 0, 1, "{:.0%}")
    axes[1].set_title(f"Human reference: agreement {c['human_reference_accuracy']:.1%}")
    axes[1].set_ylabel("Gold label")
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.set_xlabel("Label the explanation argues for (stance)")
    fig.suptitle("Does the explanation argue for the label that was predicted?", x=0.01, ha="left",
                 fontsize=11, color=INK2)
    return _save(fig, d, "fig06_consistency")


def fig_alignment(s, d, side="premise"):
    al = s["alignment_cited_vs_gold"][side]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    metrics = ["precision", "recall", "f1"]
    w = 0.8 / 3
    for mi, mname in enumerate(metrics):
        xs, ys, lo, hi = [], [], [], []
        for gi, g in enumerate(GROUPS):
            x = al[g][mname]
            if x["mean"] is None:
                continue
            xs.append(gi + (mi - 1) * w)
            ys.append(x["mean"])
            lo.append(max(x["mean"] - x["lo"], 0))
            hi.append(max(x["hi"] - x["mean"], 0))
        ax.bar(xs, ys, width=w * 0.92, color=SLOT[mi], label=mname.capitalize() if mname != "f1" else "F1",
               edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(range(len(GROUPS)), [SHORT[g] for g in GROUPS])
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Model's predicted label")
    ax.set_title(f"Cited words vs human-highlighted evidence ({side})")
    ax.legend(ncol=3, loc="upper right")
    return _save(fig, d, f"fig07_alignment_{side}")


# ------------------------------------------------------------------ tables
def write_tables(s, d):
    with open(os.path.join(d, "contrasts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "group", "n", "flip_diff", "flip_lo", "flip_hi", "drop_diff", "drop_lo", "drop_hi"])
        for name, c in s["contrasts"].items():
            for g in GROUPS:
                fd, dd = c[g]["flip_diff"], c[g]["drop_diff"]
                w.writerow([name, g, fd["n"], fd["mean"], fd["lo"], fd["hi"], dd["mean"], dd["lo"], dd["hi"]])
    with open(os.path.join(d, "variants.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "group", "n", "flip_rate", "flip_lo", "flip_hi", "prob_drop", "drop_lo", "drop_hi"])
        for k, v in s["variants"].items():
            for g in GROUPS:
                fr, pd_ = v[g]["flip_rate"], v[g]["prob_drop"]
                w.writerow([k, g, fr["n"], fr["mean"], fr["lo"], fr["hi"], pd_["mean"], pd_["lo"], pd_["hi"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    s = json.load(open(os.path.join(args.run_dir, "summary.json")))
    records = [json.loads(l) for l in open(os.path.join(args.run_dir, "records.jsonl"))]
    fdir, tdir = os.path.join(args.run_dir, "figures"), os.path.join(args.run_dir, "tables")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    paths = [fig_confusion(s, fdir), fig_removal(s, fdir, "premise"), fig_removal(s, fdir, "hypothesis"),
             fig_contrasts(s, fdir), fig_transitions(s, fdir), fig_scatter(records, fdir, "premise"),
             fig_consistency(s, fdir), fig_alignment(s, fdir, "premise"), fig_alignment(s, fdir, "hypothesis")]
    write_tables(s, tdir)
    print("Figures:\n  " + "\n  ".join(paths))
    print(f"Tables: {tdir}/contrasts.csv, {tdir}/variants.csv")


if __name__ == "__main__":
    main()

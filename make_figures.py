"""Figures and a compact table for the W2S calibration pilot (talk slide 6 and the A3 proposal).

Purpose : Turn results/metrics.json and results/test_probs.npz (written by w2s_pilot.py) into
          print-legible figures. No number is computed here that is not also derivable from
          those two files; reliability-diagram bins are recomputed from the saved probabilities.
Inputs  : results/metrics.json, results/test_probs.npz, results/curves.csv
Outputs : results/fig_pilot_reliability.png  (reliability diagrams: weak, naive W2S, conf-loss W2S, ceiling)
          results/fig_pilot_summary.png      (accuracy vs ECE per model, the "accuracy is not enough" plot)
          results/fig_pilot_curves.png       (ground-truth test accuracy per epoch; overfitting check)
          results/pilot_table.csv
Run     : python make_figures.py   (CPU is enough: needs only numpy + matplotlib)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # headless backend: write files only
import matplotlib.pyplot as plt
import numpy as np

R_DIR = Path(__file__).resolve().parent / "results"
R = json.loads((R_DIR / "metrics.json").read_text())
P = np.load(R_DIR / "test_probs.npz")
y = P["y"]
NAMES = {"weak": "Weak supervisor\n(AlexNet probe)", "naive": "W2S naive\n(DINO probe, weak labels)",
         "conf": "W2S + confidence loss\n(DINO probe, weak labels)", "ceiling": "Strong ceiling\n(DINO probe, true labels)"}
COLORS = {"weak": "#9e9e9e", "naive": "#2a7fb8", "conf": "#c0392b", "ceiling": "#222222"}
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


def rel_bins(p, y, bins=10):
    """Per-bin (mean confidence, accuracy, count) for a top-label reliability diagram."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:                     # skip near-empty bins (unstable accuracy)
            out.append((conf[m].mean(), (pred[m] == y[m]).mean(), m.sum()))
    return np.array(out)


# --- Figure A: reliability diagrams (one panel per model) ---
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
for ax, m in zip(axes, ["weak", "naive", "conf", "ceiling"]):
    b = rel_bins(P[m], y)
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1)              # perfect calibration
    ax.plot(b[:, 0], b[:, 1], "o-", c=COLORS[m], lw=1.8, ms=4)
    ax.set_title(NAMES[m], fontsize=9)
    ax.text(0.04, 0.9, f"acc {R[m]['acc']:.3f}\nECE {R[m]['ece']:.3f}", transform=ax.transAxes,
            fontsize=9, va="top", bbox=dict(fc="white", ec="0.8", lw=0.6))
    ax.set_xlabel("confidence (max prob.)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("accuracy")
fig.suptitle("CIFAR-10 pilot: reliability diagrams on the 10k test set (below the diagonal = overconfident, above = underconfident)",
             fontsize=10)
fig.tight_layout(); fig.savefig(R_DIR / "fig_pilot_reliability.png", dpi=200); plt.close(fig)

# --- Figure B: accuracy and ECE side by side ---
models = ["weak", "naive", "conf", "ceiling"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
xs = np.arange(4)
a1.bar(xs, [R[m]["acc"] * 100 for m in models], color=[COLORS[m] for m in models])
a1.set_ylabel("test accuracy (%)"); a1.set_ylim(min(R[m]["acc"] for m in models) * 100 - 5, 100)
a2.bar(xs, [R[m]["ece"] * 100 for m in models], color=[COLORS[m] for m in models])
a2.set_ylabel("ECE (%)")
for a, key in [(a1, "acc"), (a2, "ece")]:
    a.set_xticks(xs); a.set_xticklabels([NAMES[m].split("\n")[0] for m in models], fontsize=8, rotation=12)
    for x, m in zip(xs, models):
        a.text(x, R[m][key] * 100, f"{R[m][key]*100:.1f}", ha="center", va="bottom", fontsize=8)
    a.spines[["top", "right"]].set_visible(False)
a1.set_title(f"PGR: naive {R['naive']['pgr']*100:.0f}%, confidence loss {R['conf']['pgr']*100:.0f}%", fontsize=9)
a2.set_title("Calibration error (lower is better)", fontsize=9)
fig.tight_layout(); fig.savefig(R_DIR / "fig_pilot_summary.png", dpi=200); plt.close(fig)

# --- Figure C: test accuracy over epochs (full-data fits) ---
rows = list(csv.DictReader(open(R_DIR / "curves.csv")))
fig, ax = plt.subplots(figsize=(5, 3.2))
for m in ["naive", "conf", "ceiling"]:
    e = [int(r["epoch"]) for r in rows if r["method"] == m]
    a = [float(r["test_acc"]) * 100 for r in rows if r["method"] == m]
    ax.plot(e, a, c=COLORS[m], label=NAMES[m].split("\n")[0], lw=1.8)
ax.axhline(R["weak"]["acc"] * 100, c=COLORS["weak"], ls="--", label="Weak supervisor")
ax.set_xlabel("epoch"); ax.set_ylabel("ground-truth test accuracy (%)"); ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R_DIR / "fig_pilot_curves.png", dpi=200); plt.close(fig)

# --- Compact table ---
with open(R_DIR / "pilot_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "acc", "pgr", "ece", "nll", "brier", "mean_conf", "share_errors_imitated",
                "auroc_err_maxprob", "auroc_imitated_vs_correct", "conf_cov_weakcal", "conf_cov_truecal",
                "set_size_weakcal", "set_size_truecal"])
    for m in models:
        r = R[m]
        w.writerow([m, r["acc"], r.get("pgr", ""), r["ece"], r["nll"], r["brier"], r["mean_conf"],
                    r.get("share_errors_imitated", ""), r.get("auroc_err_maxprob", ""),
                    r.get("auroc_imitated_vs_correct_maxprob", ""),
                    r.get("conformal_weak_cal", {}).get("coverage_true", ""),
                    r.get("conformal_true_cal", {}).get("coverage_true", ""),
                    r.get("conformal_weak_cal", {}).get("mean_set_size", ""),
                    r.get("conformal_true_cal", {}).get("mean_set_size", "")])
print("figures written")

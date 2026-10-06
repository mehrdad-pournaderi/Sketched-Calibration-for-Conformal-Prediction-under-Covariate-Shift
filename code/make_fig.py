"""Figure 1: E1 (informed sketch, heteroscedastic model) across shift magnitude. Reads fig_hetero.json."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "results", "fig_hetero.json")))
mu = np.array(D["mu"])
series = [  # key, label, color (validated categorical slots 1-4), marker, linestyle
    ("exact weighted",        "full weighting",      "#2a78d6", "o", "-"),
    ("clipped M=20",          "clipped, $M=20$",     "#eb6834", "s", "--"),
    ("sketched k=1 weighted", "sketch $\\Pi_1$ ($k=1$)", "#1baf7a", "D", "-"),
    ("full-d unweighted",     "unweighted",          "#eda100", "^", ":"),
]
INK, MUTED, GRID = "#1f1f1e", "#5f5f5a", "#e4e4df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.titlesize": 9.5, "mathtext.fontset": "cm"})

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), constrained_layout=True)
panels = [("cov_mean", "Mean coverage", 1.0), ("cov_sd", "SD of coverage", 1.0), ("p_inf", "Infinite sets (%)", 100.0)]
for ax, (key, title, scale) in zip(axes, panels):
    for sk, lab, col, mk, ls in series:
        ax.plot(mu, scale*np.array(D[sk][key]), color=col, marker=mk, ls=ls, lw=1.6, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=3)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(r"shift magnitude $\mu$")
    ax.set_xticks(mu); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)
ax = axes[0]
ax.axhline(0.90, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.text(mu[-1]+0.05, 0.893, r"target $1-\alpha$", color=MUTED, fontsize=7.5, va="top", ha="right")
ax.set_ylim(0.55, 0.97); ax.set_yticks([0.6, 0.7, 0.8, 0.9])
axes[1].set_ylim(0, 0.08); axes[1].set_yticks([0, 0.02, 0.04, 0.06, 0.08])
axes[2].set_ylim(0, 65); axes[2].set_yticks([0, 20, 40, 60])
# secondary x-annotation: chi^2 of the full shift at each mu
ax2 = axes[2]
for m, c in zip(mu, D["chi2_X"]):
    pass
leg = axes[2].legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=2.2, labelcolor=INK, borderaxespad=0.2)
OUT = os.path.join(HERE, "..", "paper") if os.path.isdir(os.path.join(HERE, "..", "paper")) else HERE
fig.savefig(os.path.join(OUT, "fig_hetero.pdf")); fig.savefig(os.path.join(HERE, "fig_hetero.png"), dpi=180)
print("wrote", os.path.join(OUT, "fig_hetero.pdf"))
print("chi2_X:", [round(c, 1) for c in D["chi2_X"]])
print("chi2_Z1:", [round(c, 2) for c in D["chi2_Z1"]])

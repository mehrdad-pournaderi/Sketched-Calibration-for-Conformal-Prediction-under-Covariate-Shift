"""Figure 2: E7 (nonlinear multiple-index model) across the training size n. Reads exp_E7_{0..5}.json."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
runs = [json.load(open(os.path.join(HERE, "..", "results", f"exp_E7_{i}.json"))) for i in range(6)]
series = [  # rule key, label, colour (validated six-slot palette), marker, linestyle
    ("full weighting",         "full weighting",            "#2a78d6", "o", "-"),
    ("clipped M=20",           "clipped, $M=20$",           "#eb6834", "s", "--"),
    ("resid-dir k=1",          "residual sketch, $k=1$",    "#1baf7a", "D", "-"),
    ("SIR(Y) k=2 + resid-dir", "mean + residual sketch, $k=3$", "#eda100", "^", "-"),
    ("oracle [b1,b2,b3]",      "response-index oracle, $k=3$", "#e87ba4", "v", ":"),
    ("selected (n_t=100)",     "selected ($n_t=100$)",      "#008300", "P", "--"),
]
INK, MUTED, GRID = "#1f1f1e", "#5f5f5a", "#e4e4df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 9.5, "mathtext.fontset": "cm"})
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.75), constrained_layout=True)
panels = [(1.5, "cov", "Mean coverage, $\\mu=1.5$", 1.0), (2.5, "cov", "Mean coverage, $\\mu=2.5$", 1.0), (2.5, "pinf", "Infinite sets (%), $\\mu=2.5$", 100.0)]
for ax, (mu, key, title, scale) in zip(axes, panels):
    rs = sorted([r for r in runs if r["mu"] == mu], key=lambda r: r["n"]); ns = [r["n"] for r in rs]
    for rk, lab, col, mk, ls in series:
        ax.plot(ns, [scale*r["rules"][rk][key] for r in rs], color=col, marker=mk, ls=ls, lw=1.6, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_title(title, loc="left", color=INK); ax.set_xlabel("training size $n$")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)
for ax in axes[:2]:
    ax.axhline(0.90, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1); ax.set_ylim(0.78, 0.96); ax.set_yticks([0.80, 0.85, 0.90, 0.95])
axes[2].set_ylim(0, 45); axes[2].set_yticks([0, 10, 20, 30, 40])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, handlelength=2.2, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
OUT = os.path.join(HERE, "..", "paper") if os.path.isdir(os.path.join(HERE, "..", "paper")) else os.path.join(HERE, "paper")
fig.savefig(os.path.join(OUT, "fig_nonlinear.pdf")); fig.savefig(os.path.join(HERE, "fig_nonlinear.png"), dpi=180)
print("wrote", os.path.join(OUT, "fig_nonlinear.pdf"))

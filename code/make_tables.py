"""Generate the LaTeX rows for Tables E7 and E8 from the experiment JSON files (avoids transcription errors)."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
def fmt(r):
    L = "$\\infty$" if r["len"] == float("inf") or r["len"] > 1e6 else f"{r['len']:.2f}"
    return f"{r['cov']:.3f} & {r['sd']:.3f} & {L} & {100*r['pinf']:.1f}"

# ---- E7
label7 = {"full weighting": "full weighting", "clipped M=20": "clipped, $M=20$", "unweighted": "unweighted",
          "resid-dir k=1": "residual sketch, $k=1$", "resid-dir k=2 (SIR)": "residual sketch (SIR), $k=2$",
          "SIR(Y) k=2 + resid-dir": "mean (SIR on $Y$, $k=2$) $+$ residual sketch, $k=3$", "shift-dir k=1": "estimated shift direction, $k=1$",
          "random k=8": "random sketch, $k=8$", "oracle b3": "oracle noise direction $b_3$, $k=1$",
          "oracle [b1,b2,b3]": "response-index oracle $[b_1,b_2,b_3]$, $k=3$", "selected (n_t=100)": "selected by Prop.~\\ref{prop:select} ($n_t=100$)",
          "best candidate (hindsight)": "best candidate in hindsight"}
order7 = list(label7)
rows = []
for i in [1, 4, 6]:
    r = json.load(open(os.path.join(HERE, "..", "results", f"exp_E7_{i}.json")))
    d = r["diag"]
    chi = "8.5" if r["mu"] == 1.5 else "517"; ob3 = "1.25" if r["mu"] == 1.5 else "8.49"; o3 = "2.95" if r["mu"] == 1.5 else "44.3"
    head = ("\\multicolumn{5}{p{0.96\\textwidth}}{$\\mu=%s$, $n=%d$, $m=%d$, $\\chisq=%s$;\\ $\\chisqZ$: residual $k{=}1$ %.1f, mean$+$residual %.1f, random %.1f, oracle $b_3$ %s, oracle $[b_1,b_2,b_3]$ %s}\\\\"
            % (r["mu"], r["n"], r["m"], chi, d["chi2_resid1"], d["chi2_yr"], d["chi2_rand"], ob3, o3))
    rows.append(head)
    for k in order7:
        rows.append(f"{label7[k]} & {fmt(r['rules'][k])}\\\\")
    rows.append("\\addlinespace")
open(os.path.join(HERE, "..", "paper", "tab_E7_rows.tex"), "w").write("\n".join(rows[:-1]) + "\n")

# ---- E8
label8 = {"full weighting": "full weighting", "clipped M=20": "clipped, $M=20$", "unweighted": "unweighted",
          "resid-dir k=1": "residual sketch, $k=1$", "resid-dir k=2 (SIR)": "residual sketch (SIR), $k=2$",
          "SIR(Y) k=2 + resid-dir": "mean $+$ residual sketch, $k=3$", "random k=2": "random sketch, $k=2$",
          "selected (n_t=100)": "selected by Prop.~\\ref{prop:select} ($n_t=100$)", "best candidate (hindsight)": "best candidate in hindsight"}
for dsname in ["california", "protein"]:
    rows = []
    files = sorted(glob.glob(os.path.join(HERE, "..", "results", f"exp_E8_{dsname}_*.json")))
    for f in files:
        r = json.load(open(f)); d = r["diag"]
        tilt = "a random direction" if r["tilt"] == "random" else ("the feature MedInc" if r["dataset"] == "california" else "the feature F1")
        head = ("\\multicolumn{5}{p{0.96\\textwidth}}{tilt along %s, $\\gamma=%s$, $m=%d$, pool $\\chisq=%.1f$;\\ $\\hat\\chi^2_Z$ (dispersion of estimated weights): residual $k{=}1$ %.1f, SIR $k{=}2$ %.1f, mean$+$residual %.1f, random %.1f}\\\\"
                % (tilt, r["gamma"], r["m"], r["chi2"], d["chi2_resid1"], d["chi2_resid2"], d["chi2_yr"], d["chi2_rand"]))
        rows.append(head)
        for k in label8:
            rows.append(f"{label8[k]} & {fmt(r['rules'][k])}\\\\")
        rows.append("\\addlinespace")
    if rows:
        open(os.path.join(HERE, "..", "paper", f"tab_E8_{dsname}_rows.tex"), "w").write("\n".join(rows[:-1]) + "\n")
    print(dsname, "blocks:", len(files))

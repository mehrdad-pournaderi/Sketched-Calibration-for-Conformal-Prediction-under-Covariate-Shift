"""E8: real data with controlled covariate-dependent sampling (exponential tilting).
Datasets: California housing (sklearn), UCI protein tertiary structure (CASP, data/CASP.csv).
Covariates are Gaussianised by a quantile transform, winsorised at +-3, and standardised on the pool.  Target points are sampled from the
pool with probability proportional to exp(gamma <a, x>), so the ratio w(x) = exp(gamma <a,x>) / E_pool exp(gamma <a,x>) is known.
Sketched ratios w_Z = E[w | Z] are estimated by gradient-boosting regression of w on Z over pool points disjoint from the
calibration and test samples.  Predictor: gradient boosting on the full covariate.
Usage: python3 exp_E8.py {california|protein} [all | config index]
"""
import numpy as np, json, sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, '..'))
from sc_lib import *

name = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "all"
if name == "california":
    from sklearn.datasets import fetch_california_housing
    D = fetch_california_housing(); Xraw, y = D.data, np.log(D.target); feat = list(D.feature_names); key_feat = feat.index("MedInc")
else:
    df = pd.read_csv(_os.path.join(_ROOT, "data", "CASP.csv")); y = df["RMSD"].values; Xraw = df.drop(columns="RMSD").values; feat = list(df.columns[1:]); key_feat = 0
N, d = Xraw.shape
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0, subsample=N)
X = np.clip(qt.fit_transform(Xraw), -3.0, 3.0); X = (X - X.mean(0)) / X.std(0)   # Gaussianised, winsorised at +-3 sd, standardised

n_tr, ntest, n_t, R = 3000, 2000, 100, 100
CONFIGS = [("random", 1.5, 500), ("random", 1.5, 200), ("feature", 2.0, 500), ("random", 2.0, 500)] if name == "california" else [("random", 3.0, 500), ("feature", 2.0, 500)]
rng_a = np.random.default_rng(7); a_rand = rng_a.standard_normal(d); a_rand /= np.linalg.norm(a_rand)

def run_config(kind, gamma, m, seed=5):
    rng = np.random.default_rng(seed)
    a = a_rand if kind == "random" else np.eye(d)[key_feat]
    lin = X @ a; norm_c = np.mean(np.exp(gamma*lin)); wpool = np.exp(gamma*lin)/norm_c   # known ratio on the pool (target = tilted pool)
    chi2 = np.mean(wpool**2) - 1
    cand = ["full weighting", "clipped M=20", "unweighted", "resid-dir k=1", "resid-dir k=2 (SIR)", "SIR(Y) k=2 + resid-dir", "random k=2", "shift-dir (exact)"]
    names = cand + ["selected (n_t=100)", "best candidate (hindsight)"]
    res = {nm: [] for nm in names}; diag = {"chi2_resid1": [], "chi2_resid2": [], "chi2_rand": [], "chi2_yr": [], "sel_idx": []}
    for r in range(R):
        perm = rng.permutation(N)
        i_tr, i_cal, rest = perm[:n_tr], perm[n_tr:n_tr+m], perm[n_tr+m:]
        i_pool, pool_t = rest[:len(rest)//2], rest[len(rest)//2:]          # unlabeled pool for w_Z; pool from which the tilted target is drawn
        p = wpool[pool_t] / wpool[pool_t].sum()
        i_te = rng.choice(pool_t, size=ntest, replace=True, p=p)            # target test sample (tilted, with replacement)
        i_t = rng.choice(pool_t, size=n_t, replace=True, p=p)               # labelled target sample for selection
        Xtr, Ytr = X[i_tr], y[i_tr]; Xcal, Ycal = X[i_cal], y[i_cal]; Xte, Yte = X[i_te], y[i_te]; Xt, Yt = X[i_t], y[i_t]
        h = n_tr//2
        gb1 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr[:h], Ytr[:h])
        gb2 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr[h:], Ytr[h:])
        Rcf = np.abs(np.r_[Ytr[:h]-gb2.predict(Xtr[:h]), Ytr[h:]-gb1.predict(Xtr[h:])])
        gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr)
        A = np.c_[Xtr, np.ones(n_tr)]; c, *_ = np.linalg.lstsq(A, Rcf, rcond=None); v = c[:d]/np.linalg.norm(c[:d])
        Pi_r1 = v[None, :]; Pi_r2 = sir_directions(Xtr, Rcf, k=2, H=10); Pi_rand = orth_rows(rng.standard_normal((2, d)))
        Pi_yr = orth_rows(np.vstack([sir_directions(Xtr, Ytr, k=2, H=10), v]))
        def wZ_est(Pi):      # E[w | Pi x] by regression of the known ratio on the sketch over the disjoint pool
            reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[i_pool]@Pi.T, wpool[i_pool])
            return lambda x: np.maximum(reg.predict(x@Pi.T), 1e-6)
        s_cal = np.abs(Ycal - gb.predict(Xcal)); s_te = np.abs(Yte - gb.predict(Xte)); s_t = np.abs(Yt - gb.predict(Xt))
        wf = lambda x: np.exp(gamma*(x@a))/norm_c; wc = lambda x: np.minimum(wf(x), 20.0)
        rules = {"full weighting": wf, "clipped M=20": wc, "unweighted": lambda x: np.ones(len(x)),
                 "resid-dir k=1": wZ_est(Pi_r1), "resid-dir k=2 (SIR)": wZ_est(Pi_r2), "SIR(Y) k=2 + resid-dir": wZ_est(Pi_yr), "random k=2": wZ_est(Pi_rand),
                 "shift-dir (exact)": wf}       # the sketch Z=<a,x> carries the whole shift: w_Z = w
        out_r = {}; sel_cov = {}
        for nm, wfun in rules.items():
            out_r[nm] = evaluate(s_cal, wfun(Xcal), s_te, wfun(Xte)); res[nm].append(out_r[nm])
            q_t = wq_threshold(s_cal, wfun(Xcal), wfun(Xt), 1-alpha); sel_cov[nm] = np.mean(s_t <= q_t)
        j = min(cand, key=lambda nm: abs(sel_cov[nm]-(1-alpha))); res["selected (n_t=100)"].append(out_r[j]); diag["sel_idx"].append(cand.index(j))
        jb = min(cand, key=lambda nm: abs(out_r[nm][0]-(1-alpha))); res["best candidate (hindsight)"].append(out_r[jb])
        for key, Pi in (("chi2_resid1", Pi_r1), ("chi2_resid2", Pi_r2), ("chi2_rand", Pi_rand), ("chi2_yr", Pi_yr)):
            wz = rules[{"chi2_resid1": "resid-dir k=1", "chi2_resid2": "resid-dir k=2 (SIR)", "chi2_rand": "random k=2", "chi2_yr": "SIR(Y) k=2 + resid-dir"}[key]](X[i_pool])
            diag[key].append(np.mean(wz**2)/np.mean(wz)**2 - 1)
    tag = f"{name}, tilt={kind}, gamma={gamma}, m={m}"
    print(f"=== E8 {tag}, R={R}: pool chi2={chi2:.2f}  chi2_Z resid1 {np.mean(diag['chi2_resid1']):.2f} resid2 {np.mean(diag['chi2_resid2']):.2f} random2 {np.mean(diag['chi2_rand']):.2f} SIR(Y)+resid {np.mean(diag['chi2_yr']):.2f}"
          f" | selection counts {np.bincount(diag['sel_idx'], minlength=len(cand)).tolist()}", flush=True)
    for nm in names: print(report(nm, res[nm]), flush=True)
    return {"dataset": name, "tilt": kind, "gamma": gamma, "m": m, "chi2": float(chi2), "cand": cand,
            "diag": {k: (float(np.mean(v)) if k != "sel_idx" else np.bincount(v, minlength=len(cand)).tolist()) for k, v in diag.items()},
            "rules": {nm: {"cov": float(np.mean(np.array(v)[:,0])), "sd": float(np.std(np.array(v)[:,0])), "len": float(np.median(np.array(v)[:,1])),
                           "pinf": float(np.mean(np.array(v)[:,2])), "plow": float(np.mean(np.array(v)[:,0]<0.85))} for nm, v in res.items()}}

idx = range(len(CONFIGS)) if which == "all" else [int(which)]
for i in idx:
    out = run_config(*CONFIGS[i])
    json.dump(out, open(_os.path.join(_ROOT, "results", f"exp_E8_{name}_{i}.json"), "w"), indent=1)

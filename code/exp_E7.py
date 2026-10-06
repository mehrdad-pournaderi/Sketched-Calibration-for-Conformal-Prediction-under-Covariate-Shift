"""E7: nonlinear multiple-index model, gradient-boosting predictor on the full covariate, sketched weights.
Y = 2 sin<b1,X> + 0.5 <b2,X>^2 + sigma(<b3,X>) xi,  sigma(t) = 0.5 + 0.5 t_+,  X ~ N(0, I_40) (source), N(mu a, I_40) (target),
a = 0.6 b3 + 0.5 b1 + sqrt(0.39) g (g random, orthogonal to b1,b2,b3).  Sketches are learned from the training fold;
the shift direction is estimated from unlabeled covariates; a labelled target sample of size n_t selects among candidates.
Usage: python3 exp_E7.py [config index 0..6 | all]
"""
import numpy as np, json, sys, os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, '..'))
from sc_lib import *

d, ntest, nu, n_t = 40, 20000, 2000, 100
rng0 = np.random.default_rng(0)
B = orth_rows(rng0.standard_normal((3, d))); b1, b2, b3 = B
g0 = rng0.standard_normal(d); g0 -= B.T@(B@g0); g0 /= np.linalg.norm(g0)
a = 0.6*b3 + 0.5*b1 + np.sqrt(0.39)*g0
def mean_fn(X): return 2*np.sin(X@b1) + 0.5*(X@b2)**2
def sfun(X): return 0.5 + 0.5*np.maximum(0.0, X@b3)
def wexp(pa, mu): return lambda z: np.exp(mu*(z@pa) - mu**2*(pa@pa)/2)
w_one = lambda z: np.ones(len(z))
def fit_ols_dir(X, r):
    A = np.c_[X, np.ones(len(X))]; c, *_ = np.linalg.lstsq(A, r, rcond=None); v = c[:d]; return v/np.linalg.norm(v)

CONFIGS = [(1.5, 1000, 500), (1.5, 4000, 500), (1.5, 16000, 500), (2.5, 1000, 500), (2.5, 4000, 500), (2.5, 16000, 500), (2.5, 4000, 200)]

def run_config(mu, n, m, R=100, seed=31):
    rng = np.random.default_rng(seed)
    cand = ["full weighting", "clipped M=20", "unweighted", "resid-dir k=1", "resid-dir k=2 (SIR)", "SIR(Y) k=2 + resid-dir", "shift-dir k=1", "resid+shift k=2", "random k=8"]
    names = cand + ["oracle b3", "oracle [b1,b2,b3]", "selected (n_t=100)", "best candidate (hindsight)"]
    res = {nm: [] for nm in names}; diag = {"sin_resid_b3": [], "sin_shift_a": [], "chi2_resid1": [], "chi2_shift": [], "chi2_rs": [], "chi2_rand": [], "chi2_yr": [], "sel_idx": []}
    for r in range(R):
        Xtr = rng.standard_normal((n, d)); Ytr = mean_fn(Xtr) + sfun(Xtr)*rng.standard_normal(n)
        Xcal = rng.standard_normal((m, d)); Ycal = mean_fn(Xcal) + sfun(Xcal)*rng.standard_normal(m)
        Xte = mu*a + rng.standard_normal((ntest, d)); Yte = mean_fn(Xte) + sfun(Xte)*rng.standard_normal(ntest)
        Xt = mu*a + rng.standard_normal((n_t, d)); Yt = mean_fn(Xt) + sfun(Xt)*rng.standard_normal(n_t)      # labelled target sample for selection
        Xu_s = rng.standard_normal((nu, d)); Xu_t = mu*a + rng.standard_normal((nu, d))                   # unlabeled covariates
        # predictor on the full covariate (training fold), cross-fitted residuals for the sketch
        h = n//2
        gb1 = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr[:h], Ytr[:h])
        gb2 = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr[h:], Ytr[h:])
        Rcf = np.abs(np.r_[Ytr[:h]-gb2.predict(Xtr[:h]), Ytr[h:]-gb1.predict(Xtr[h:])])
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(Xtr, Ytr)
        v_res = fit_ols_dir(Xtr, Rcf); Pi_r1 = v_res[None, :]; Pi_r2 = sir_directions(Xtr, Rcf, k=2, H=10)
        clf = LogisticRegression(C=1e4, max_iter=2000).fit(np.vstack([Xu_s, Xu_t]), np.r_[np.zeros(nu), np.ones(nu)])
        v_sh = clf.coef_.ravel(); v_sh /= np.linalg.norm(v_sh); Pi_sh = v_sh[None, :]
        Pi_rs = orth_rows(np.vstack([v_res, v_sh])); Pi_rand = orth_rows(rng.standard_normal((8, d))); Pi_or = b3[None, :]
        Pi_y = sir_directions(Xtr, Ytr, k=2, H=10); Pi_yr = orth_rows(np.vstack([Pi_y, v_res])); Pi_or3 = B
        s_cal = np.abs(Ycal - gb.predict(Xcal)); s_te = np.abs(Yte - gb.predict(Xte)); s_t = np.abs(Yt - gb.predict(Xt))
        wf = wexp(a, mu); wc = lambda x: np.minimum(wf(x), 20.0)
        rules = {"full weighting": (np.eye(d), wf), "clipped M=20": (np.eye(d), wc), "unweighted": (np.eye(d), w_one),
                 "resid-dir k=1": (Pi_r1, wexp(Pi_r1@a, mu)), "resid-dir k=2 (SIR)": (Pi_r2, wexp(Pi_r2@a, mu)),
                 "shift-dir k=1": (Pi_sh, wexp(Pi_sh@a, mu)), "resid+shift k=2": (Pi_rs, wexp(Pi_rs@a, mu)),
                 "SIR(Y) k=2 + resid-dir": (Pi_yr, wexp(Pi_yr@a, mu)),
                 "random k=8": (Pi_rand, wexp(Pi_rand@a, mu)), "oracle b3": (Pi_or, wexp(Pi_or@a, mu)), "oracle [b1,b2,b3]": (Pi_or3, wexp(Pi_or3@a, mu))}
        out_r = {}; sel_cov = {}
        for nm, (Pi, wfun) in rules.items():
            out_r[nm] = evaluate(s_cal, wfun(Xcal@Pi.T), s_te, wfun(Xte@Pi.T)); res[nm].append(out_r[nm])
            if nm in cand:
                q_t = wq_threshold(s_cal, wfun(Xcal@Pi.T), wfun(Xt@Pi.T), 1-alpha); sel_cov[nm] = np.mean(s_t <= q_t)
        j = min(cand, key=lambda nm: abs(sel_cov[nm]-(1-alpha))); res["selected (n_t=100)"].append(out_r[j]); diag["sel_idx"].append(cand.index(j))
        jb = min(cand, key=lambda nm: abs(out_r[nm][0]-(1-alpha))); res["best candidate (hindsight)"].append(out_r[jb])
        diag["sin_resid_b3"].append(np.sqrt(max(0.0, 1-(v_res@b3)**2))); diag["sin_shift_a"].append(np.sqrt(max(0.0, 1-(v_sh@a)**2)))
        for key, Pi in (("chi2_resid1", Pi_r1), ("chi2_shift", Pi_sh), ("chi2_rs", Pi_rs), ("chi2_rand", Pi_rand), ("chi2_yr", Pi_yr)):
            diag[key].append(np.exp(mu**2*((Pi@a)@(Pi@a)))-1)
    tag = f"mu={mu}, n={n}, m={m}"
    print(f"=== E7 {tag}, R={R}: chi2_X={np.exp(mu**2)-1:.1f}  chi2_Z resid1 {np.mean(diag['chi2_resid1']):.2f} shift {np.mean(diag['chi2_shift']):.2f} resid+shift {np.mean(diag['chi2_rs']):.2f} random8 {np.mean(diag['chi2_rand']):.2f} SIR(Y)+resid {np.mean(diag['chi2_yr']):.2f} oracle b3 {np.exp(mu**2*0.36)-1:.2f} oracle 3 {np.exp(mu**2*0.61)-1:.2f}"
          f" | sin(resid-dir,b3) {np.mean(diag['sin_resid_b3']):.3f}  sin(shift-dir,a) {np.mean(diag['sin_shift_a']):.3f} | selection counts {np.bincount(diag['sel_idx'], minlength=len(cand)).tolist()}", flush=True)
    for nm in names: print(report(nm, res[nm]), flush=True)
    return {"mu": mu, "n": n, "m": m, "diag": {k: (float(np.mean(v)) if k != "sel_idx" else np.bincount(v, minlength=len(cand)).tolist()) for k, v in diag.items()}, "cand": cand,
            "rules": {nm: {"cov": float(np.mean(np.array(v)[:,0])), "sd": float(np.std(np.array(v)[:,0])), "len": float(np.median(np.array(v)[:,1])),
                           "pinf": float(np.mean(np.array(v)[:,2])), "plow": float(np.mean(np.array(v)[:,0]<0.85))} for nm, v in res.items()}}

which = sys.argv[1] if len(sys.argv) > 1 else "all"
idx = range(len(CONFIGS)) if which == "all" else [int(which)]
for i in idx:
    out = run_config(*CONFIGS[i])
    json.dump(out, open(_os.path.join(_ROOT, "results", f"exp_E7_{i}.json"), "w"), indent=1)

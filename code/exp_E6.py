"""E6: full-covariate score with sketched weights, Gaussian single-index model.
(a) heteroscedastic setting of E1 (seed 11), mu in {1.5, 2.5}; (b) homoscedastic setting of E3 (seed 5, random sketch k=8), mu=1.5.
Writes exp_E6.json and prints a log.
"""
import numpy as np, json, sys
from scipy.stats import norm
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, '..'))
from sc_lib import *

d, n, m, ntest = 40, 2000, 500, 20000
rng0 = np.random.default_rng(0)
beta = rng0.standard_normal(d); beta *= 3/np.linalg.norm(beta); bdir = beta/np.linalg.norm(beta)

def fit_ols(X, Y):
    A = np.c_[X, np.ones(len(X))]; coef, *_ = np.linalg.lstsq(A, Y, rcond=None); return coef
def pred(coef, X): return np.c_[X, np.ones(len(X))] @ coef
def wexp(pa, mu): return lambda z: np.exp(mu*(z@pa) - mu**2*(pa@pa)/2)
w_one = lambda z: np.ones(len(z))

def run_sketched_score(Pi, wfun, Xtr, Ytr, Xcal, Ycal, Xte, Yte):
    Ztr, Zcal, Zte = Xtr@Pi.T, Xcal@Pi.T, Xte@Pi.T
    c = fit_ols(Ztr, Ytr)
    return evaluate(np.abs(Ycal-pred(c, Zcal)), wfun(Zcal), np.abs(Yte-pred(c, Zte)), wfun(Zte))

def run_full_score(Pi, wfun, coef_full, Xcal, Ycal, Xte, Yte):
    Zcal, Zte = Xcal@Pi.T, Xte@Pi.T
    return evaluate(np.abs(Ycal-pred(coef_full, Xcal)), wfun(Zcal), np.abs(Yte-pred(coef_full, Xte)), wfun(Zte))

out = {}
R = 200
def setting(tag, hetero, mu, seed, shift_fn, R=R):
    rng = np.random.default_rng(seed)
    a = shift_fn(rng)
    sfun = (lambda X: 0.5+0.5*np.maximum(0.0, X@bdir)) if hetero else (lambda X: np.ones(len(X)))
    Pi_rand = orth_rows(rng.standard_normal((8, d)))      # fixed random sketch for this setting (as in E3)
    names = ["full weighting", "unweighted", "sketched score, Pi_1", "sketched score, random k=8",
             "full score, weights Pi_1", "full score, weights Pi_8", "full score, weights resid-dir", "full score, weights random k=8"]
    res = {nm: [] for nm in names}; diag = {"chi2_Z1": [], "chi2_SIR": [], "chi2_rand": [], "leak_full_rand_bound": [], "sin_sir": [], "sin_ols": []}
    for r in range(R):
        Xtr = rng.standard_normal((n, d)); Ytr = Xtr@beta + sfun(Xtr)*rng.standard_normal(n)
        Xcal = rng.standard_normal((m, d)); Ycal = Xcal@beta + sfun(Xcal)*rng.standard_normal(m)
        Xte = mu*a + rng.standard_normal((ntest, d)); Yte = Xte@beta + sfun(Xte)*rng.standard_normal(ntest)
        coef = fit_ols(Xtr, Ytr); theta = coef[:d]; bh = theta/np.linalg.norm(theta)
        Pi1 = bh[None, :]; Pi8 = orth_rows(np.vstack([bh, rng.standard_normal((7, d))]))
        # residual sketch (Section 4.9): cross-fitted absolute residuals on the whole fold, least-squares direction of |R| on X
        h = n//2; c1 = fit_ols(Xtr[:h], Ytr[:h]); c2 = fit_ols(Xtr[h:], Ytr[h:])
        Rcf = np.abs(np.r_[Ytr[:h] - pred(c2, Xtr[:h]), Ytr[h:] - pred(c1, Xtr[h:])])
        vres = fit_ols(Xtr, Rcf)[:d]; Pi_sir = (vres/np.linalg.norm(vres))[None, :]
        wf = wexp(a, mu)
        res["full weighting"].append(run_full_score(np.eye(d), wf, coef, Xcal, Ycal, Xte, Yte))
        res["unweighted"].append(run_full_score(np.eye(d), w_one, coef, Xcal, Ycal, Xte, Yte))
        res["sketched score, Pi_1"].append(run_sketched_score(Pi1, wexp(Pi1@a, mu), Xtr, Ytr, Xcal, Ycal, Xte, Yte))
        res["sketched score, random k=8"].append(run_sketched_score(Pi_rand, wexp(Pi_rand@a, mu), Xtr, Ytr, Xcal, Ycal, Xte, Yte))
        res["full score, weights Pi_1"].append(run_full_score(Pi1, wexp(Pi1@a, mu), coef, Xcal, Ycal, Xte, Yte))
        res["full score, weights Pi_8"].append(run_full_score(Pi8, wexp(Pi8@a, mu), coef, Xcal, Ycal, Xte, Yte))
        res["full score, weights resid-dir"].append(run_full_score(Pi_sir, wexp(Pi_sir@a, mu), coef, Xcal, Ycal, Xte, Yte))
        res["full score, weights random k=8"].append(run_full_score(Pi_rand, wexp(Pi_rand@a, mu), coef, Xcal, Ycal, Xte, Yte))
        diag["chi2_Z1"].append(np.exp(mu**2*((Pi1@a)@(Pi1@a)))-1); diag["chi2_SIR"].append(np.exp(mu**2*((Pi_sir@a)@(Pi_sir@a)))-1)
        diag["chi2_rand"].append(np.exp(mu**2*((Pi_rand@a)@(Pi_rand@a)))-1)
        diag["sin_sir"].append(np.sqrt(max(0.0, 1-(Pi_sir[0]@bdir)**2))); diag["sin_ols"].append(np.sqrt(max(0.0, 1-(bh@bdir)**2)))
        # homoscedastic full-score leakage bound for the random sketch: mu |<(beta-theta)_perp, a_perp>| / (sqrt(2 pi) s_theta), s_theta^2 = 1 + |(beta-theta)_perp|^2
        dlt = beta-theta; dp = dlt - Pi_rand.T@(Pi_rand@dlt); ap = a - Pi_rand.T@(Pi_rand@a)
        diag["leak_full_rand_bound"].append(2*norm.cdf(mu*abs(dp@ap)/(2*np.sqrt(1+dp@dp)))-1)
    print(f"=== E6{tag}: {'heteroscedastic' if hetero else 'homoscedastic'}, mu={mu}, R={R}", flush=True)
    print(f"  chi2_X={np.exp(mu**2)-1:.1f}  chi2_Z: Pi_1 {np.mean(diag['chi2_Z1']):.2f}  resid-dir {np.mean(diag['chi2_SIR']):.2f}  random8 {np.mean(diag['chi2_rand']):.2f}"
          f"  sin(angle) OLS {np.mean(diag['sin_ols']):.3f} resid-dir {np.mean(diag['sin_sir']):.3f}  full-score random-sketch leakage (homosc. formula) {np.mean(diag['leak_full_rand_bound']):.4f}", flush=True)
    for nm in names: print(report(nm, res[nm]), flush=True)
    out[tag] = {"mu": mu, "hetero": hetero, "diag": {k: float(np.mean(v)) for k, v in diag.items()},
                "rules": {nm: {"cov": float(np.mean(np.array(v)[:,0])), "sd": float(np.std(np.array(v)[:,0])), "len": float(np.median(np.array(v)[:,1])),
                               "pinf": float(np.mean(np.array(v)[:,2])), "plow": float(np.mean(np.array(v)[:,0]<0.85))} for nm, v in res.items()}}
    json.dump(out, open(_os.path.join(_ROOT, "results", "exp_E6.json"), "w"), indent=1)

def shift_hetero(rng):
    g = rng.standard_normal(d); g -= (g@bdir)*bdir; g /= np.linalg.norm(g); return 0.7*bdir+np.sqrt(0.51)*g
def shift_random(rng):
    a = rng.standard_normal(d); return a/np.linalg.norm(a)

which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("a", "all"):
    setting("a15", True, 1.5, 11, shift_hetero); setting("a25", True, 2.5, 11, shift_hetero)
if which in ("b", "all"):
    setting("b15", False, 1.5, 5, shift_random)

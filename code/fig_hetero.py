"""E1 (heteroscedastic model, informed sketches) on a finer shift grid, for Figure 1 of the draft.
Same seeds/model as E1 in sketched_calibration_experiments.py (rng re-seeded per mu, so
values at mu in {1.0, 1.5, 2.5} coincide with Table 1). Writes fig_hetero.json.
"""
import numpy as np, json, os
from scipy.stats import norm

d, n, m, ntest, alpha = 40, 2000, 500, 20000, 0.1
rng0 = np.random.default_rng(0)
beta = rng0.standard_normal(d); beta *= 3/np.linalg.norm(beta)
bdir = beta/np.linalg.norm(beta)

def orth_rows(M):
    q,_ = np.linalg.qr(M.T); return q.T
def fit_predict(Ztr, Ytr, Z):
    A = np.c_[Ztr, np.ones(len(Ztr))]; coef,*_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return np.c_[Z, np.ones(len(Z))] @ coef
def wq_threshold(scores, w_cal, w_test, level):
    order = np.argsort(scores); s, w = scores[order], w_cal[order]
    cw = np.cumsum(w); tot = cw[-1]
    idx = np.searchsorted(cw, level*(tot+w_test))
    return np.where(idx < len(s), s[np.minimum(idx, len(s)-1)], np.inf)
def run(Pi, wfun, Xtr, Ytr, Xcal, Ycal, Xte, Yte):
    Ztr, Zcal, Zte = Xtr@Pi.T, Xcal@Pi.T, Xte@Pi.T
    pc = fit_predict(Ztr, Ytr, Zcal); pt = fit_predict(Ztr, Ytr, Zte)
    q = wq_threshold(np.abs(Ycal-pc), wfun(Zcal), wfun(Zte), 1-alpha)
    return np.mean(np.abs(Yte-pt) <= q), np.median(2*q), np.mean(np.isinf(q))
def wexp(pa, mu):
    return lambda z: np.exp(mu*(z@pa) - mu**2*(pa@pa)/2)
w_one = lambda z: np.ones(len(z))

R = 200
sfun = lambda X: 0.5+0.5*np.maximum(0.0, X@bdir)
names = ["exact weighted","clipped M=20","sketched k=1 weighted","sketched k=8 weighted","full-d unweighted"]
out = {"mu": [], "chi2_X": [], "chi2_Z1": [], "chi2_Z8": []}
for nm in names: out[nm] = {"cov_mean": [], "cov_sd": [], "p_inf": [], "med_len": [], "p_low": []}
for mu in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    rng = np.random.default_rng(11)
    g = rng.standard_normal(d); g -= (g@bdir)*bdir; g /= np.linalg.norm(g); a = 0.7*bdir+np.sqrt(0.51)*g
    res = {nm:[] for nm in names}; c1, c8 = [], []
    for r in range(R):
        Xtr = rng.standard_normal((n,d)); Ytr = Xtr@beta + sfun(Xtr)*rng.standard_normal(n)
        Xcal = rng.standard_normal((m,d)); Ycal = Xcal@beta + sfun(Xcal)*rng.standard_normal(m)
        Xte = mu*a + rng.standard_normal((ntest,d)); Yte = Xte@beta + sfun(Xte)*rng.standard_normal(ntest)
        bh,*_ = np.linalg.lstsq(np.c_[Xtr,np.ones(n)], Ytr, rcond=None); bh = bh[:d]/np.linalg.norm(bh[:d])
        Pi1 = bh[None,:]; Pi8 = orth_rows(np.vstack([bh, rng.standard_normal((7,d))]))
        c1.append(np.exp(mu**2*((Pi1@a)@(Pi1@a)))-1); c8.append(np.exp(mu**2*((Pi8@a)@(Pi8@a)))-1)
        wf = wexp(a,mu); wc = lambda x: np.minimum(wf(x),20.0)
        res["exact weighted"].append(run(np.eye(d), wf, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["clipped M=20"].append(run(np.eye(d), wc, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["sketched k=1 weighted"].append(run(Pi1, wexp(Pi1@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["sketched k=8 weighted"].append(run(Pi8, wexp(Pi8@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["full-d unweighted"].append(run(np.eye(d), w_one, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
    out["mu"].append(mu); out["chi2_X"].append(np.exp(mu**2)-1)
    out["chi2_Z1"].append(float(np.mean(c1))); out["chi2_Z8"].append(float(np.mean(c8)))
    for nm in names:
        v = np.array(res[nm])
        out[nm]["cov_mean"].append(float(v[:,0].mean())); out[nm]["cov_sd"].append(float(v[:,0].std()))
        out[nm]["p_inf"].append(float(v[:,2].mean())); out[nm]["med_len"].append(float(np.median(v[:,1])))
        out[nm]["p_low"].append(float(np.mean(v[:,0]<0.85)))
        print(f"mu={mu} {nm:24s} cov {v[:,0].mean():.3f} sd {v[:,0].std():.3f} len {np.median(v[:,1]):.2f} Pinf {v[:,2].mean():.3f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "fig_hetero.json"),"w"), indent=1)

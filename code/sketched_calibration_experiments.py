"""Experiments for "Sketched Calibration for Conformal Prediction under Covariate Shift".
Usage: python3 sketched_calibration_experiments.py {E1|E2|E3|E4|E5|all}
  E1: informed sketches (k=1, k=8), heteroscedastic model, shift with component 0.7 along the index (Table 1).
  E2: estimated ratios by logistic regression on unlabeled covariates, full-d vs sketched (Table 2).
  E3: random sketch k=8, random shift direction, homoscedastic model (Table 3).
  E4: shift along the signal, sweep over sketch dimension k (Table 4).
  E5: informed sketches, homoscedastic model, random shift direction (control; Table 5).
Figure 1 (E1 on a finer shift grid) is produced by fig_hetero.py / make_fig.py.
"""
import numpy as np, sys
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

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
    cw = np.cumsum(w); tot = cw[-1]; out = np.empty(len(w_test))
    for j, wt in enumerate(w_test):
        idx = np.searchsorted(cw, level*(tot+wt)); out[j] = s[idx] if idx < len(s) else np.inf
    return out
def run(Pi, wfun, Xtr, Ytr, Xcal, Ycal, Xte, Yte):
    Ztr, Zcal, Zte = Xtr@Pi.T, Xcal@Pi.T, Xte@Pi.T
    pc = fit_predict(Ztr, Ytr, Zcal); pt = fit_predict(Ztr, Ytr, Zte)
    q = wq_threshold(np.abs(Ycal-pc), wfun(Zcal), wfun(Zte), 1-alpha)
    return np.mean(np.abs(Yte-pt) <= q), np.median(2*q), np.mean(np.isinf(q))
def wexp(pa, mu):            # exact ratio in sketched coords, pa = Pi a
    return lambda z: np.exp(mu*(z@pa) - mu**2*(pa@pa)/2)
w_one = lambda z: np.ones(len(z))
def report(name, v, extra=""):
    v = np.array(v)
    print(f"  {name:34s} cov {v[:,0].mean():.3f}  sd {v[:,0].std():.3f}  P(cov<.85) {np.mean(v[:,0]<.85):.3f}  len {np.median(v[:,1]):6.2f}  P(inf) {v[:,2].mean():.3f} {extra}")

which = sys.argv[1] if len(sys.argv)>1 else "all"

if which in ("E3","all"):
    print("=== E3: random sketch k=8, random shift, mu=1.5, sigma=1, R=300")
    rng = np.random.default_rng(5); mu, sigma, k = 1.5, 1.0, 8
    a = rng.standard_normal(d); a /= np.linalg.norm(a)
    Pi = orth_rows(rng.standard_normal((k,d))); a_par = Pi.T@(Pi@a); a_perp = a-a_par
    Pi_aw = orth_rows(np.vstack([Pi, a_perp/np.linalg.norm(a_perp)]))
    b_perp = beta - Pi.T@(Pi@beta); s2 = sigma**2 + b_perp@b_perp
    dY = 2*norm.cdf(mu*abs(b_perp@a_perp)/(2*np.sqrt(s2)))-1
    print(f"  |a_par|^2={a_par@a_par:.3f} |beta_perp|^2={b_perp@b_perp:.3f} chi2_X={np.exp(mu**2)-1:.2f} chi2_Z={np.exp(mu**2*(a_par@a_par))-1:.3f} leakage={dY:.4f} 1-a-leak={1-alpha-dY:.4f}")
    print(f"  oracle lengths: full {2*1.645*sigma:.2f}, sketched {2*1.645*np.sqrt(s2):.2f}")
    res = {k_:[] for k_ in ["exact weighted (Pi=I)","sketched weighted k=8","shift-aware sketch k=9","sketched unweighted","full-d unweighted"]}
    for r in range(300):
        Xtr = rng.standard_normal((n,d)); Ytr = Xtr@beta + sigma*rng.standard_normal(n)
        Xcal = rng.standard_normal((m,d)); Ycal = Xcal@beta + sigma*rng.standard_normal(m)
        Xte = mu*a + rng.standard_normal((ntest,d)); Yte = Xte@beta + sigma*rng.standard_normal(ntest)
        res["exact weighted (Pi=I)"].append(run(np.eye(d), wexp(a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["sketched weighted k=8"].append(run(Pi, wexp(Pi@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["shift-aware sketch k=9"].append(run(Pi_aw, wexp(Pi_aw@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["sketched unweighted"].append(run(Pi, w_one, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        res["full-d unweighted"].append(run(np.eye(d), w_one, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
    for nm,v in res.items(): report(nm, v)

if which in ("E4","all"):
    print("=== E4: shift along signal a=beta/|beta|, mu=1.5, sigma=1, R=150")
    mu, sigma = 1.5, 1.0; a = bdir
    for k in [2,5,10,20,30,40]:
        rng = np.random.default_rng(1)
        Pi = orth_rows(rng.standard_normal((k,d))) if k<d else np.eye(d)
        a_par = Pi.T@(Pi@a); a_perp = a-a_par; b_perp = beta-Pi.T@(Pi@beta); s2 = sigma**2+b_perp@b_perp
        dY = 2*norm.cdf(mu*abs(b_perp@a_perp)/(2*np.sqrt(s2)))-1
        rs, rf = [], []
        for r in range(150):
            Xtr = rng.standard_normal((n,d)); Ytr = Xtr@beta + sigma*rng.standard_normal(n)
            Xcal = rng.standard_normal((m,d)); Ycal = Xcal@beta + sigma*rng.standard_normal(m)
            Xte = mu*a + rng.standard_normal((ntest,d)); Yte = Xte@beta + sigma*rng.standard_normal(ntest)
            rs.append(run(Pi, wexp(Pi@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            rf.append(run(np.eye(d), wexp(a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        rs, rf = np.array(rs), np.array(rf)
        print(f"  k={k:2d} |a_par|^2={a_par@a_par:.3f} chi2_Z={np.exp(mu**2*(a_par@a_par))-1:5.2f} leak={dY:.3f} 1-a-leak={1-alpha-dY:.3f} | sk: cov {rs[:,0].mean():.3f} sd {rs[:,0].std():.3f} len {np.median(rs[:,1]):5.2f} Pinf {rs[:,2].mean():.3f} | full: cov {rf[:,0].mean():.3f} sd {rf[:,0].std():.3f} len {np.median(rf[:,1]):5.2f} Pinf {rf[:,2].mean():.3f}")

def E3(hetero, R=200):
    tag = "E1: heteroscedastic, <a,bdir>=0.7" if hetero else "E5: homoscedastic sigma=1, random a"
    print(f"=== {tag}, R={R}")
    sfun = (lambda X: 0.5+0.5*np.maximum(0.0, X@bdir)) if hetero else (lambda X: np.ones(len(X)))
    for mu in [1.0,1.5,2.5]:
        rng = np.random.default_rng(7 if not hetero else 11)
        if hetero:
            g = rng.standard_normal(d); g -= (g@bdir)*bdir; g /= np.linalg.norm(g); a = 0.7*bdir+np.sqrt(0.51)*g
        else:
            a = rng.standard_normal(d); a /= np.linalg.norm(a)
        names = ["exact weighted","clipped M=20","sketched k=1 weighted","sketched k=8 weighted","sketched k=8 unweighted","full-d unweighted"]
        res = {nm:[] for nm in names}; c1, c8, l1, l8 = [], [], [], []
        for r in range(R):
            Xtr = rng.standard_normal((n,d)); Ytr = Xtr@beta + sfun(Xtr)*rng.standard_normal(n)
            Xcal = rng.standard_normal((m,d)); Ycal = Xcal@beta + sfun(Xcal)*rng.standard_normal(m)
            Xte = mu*a + rng.standard_normal((ntest,d)); Yte = Xte@beta + sfun(Xte)*rng.standard_normal(ntest)
            bh,*_ = np.linalg.lstsq(np.c_[Xtr,np.ones(n)], Ytr, rcond=None); bh = bh[:d]/np.linalg.norm(bh[:d])
            Pi1 = bh[None,:]; Pi8 = orth_rows(np.vstack([bh, rng.standard_normal((7,d))]))
            for Pi, cc, ll in ((Pi1,c1,l1),(Pi8,c8,l8)):
                pa = Pi@a; cc.append(np.exp(mu**2*(pa@pa))-1)
                bp = beta-Pi.T@(Pi@beta); ap = a-Pi.T@pa; s2 = 1.0+bp@bp
                ll.append(2*norm.cdf(mu*abs(bp@ap)/(2*np.sqrt(s2)))-1)
            wf = wexp(a,mu); wc = lambda x: np.minimum(wf(x),20.0)
            res["exact weighted"].append(run(np.eye(d), wf, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            res["clipped M=20"].append(run(np.eye(d), wc, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            res["sketched k=1 weighted"].append(run(Pi1, wexp(Pi1@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            res["sketched k=8 weighted"].append(run(Pi8, wexp(Pi8@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            res["sketched k=8 unweighted"].append(run(Pi8, w_one, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            res["full-d unweighted"].append(run(np.eye(d), w_one, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
        print(f" mu={mu}: chi2_X={np.exp(mu**2)-1:.1f}  chi2_Z(k=1)={np.mean(c1):.2f}  chi2_Z(k=8)={np.mean(c8):.2f}  homosc-leak k=1 {np.mean(l1):.4f} k=8 {np.mean(l8):.4f}")
        for nm,v in res.items(): report(nm, v)

if which in ("E1","all"): E3(True)
if which in ("E5","all"): E3(False)

if which in ("E2","all"):
    print("=== E2: estimated weights via logistic regression on unlabeled covariates (n_u=2000 each), heteroscedastic model, R=200")
    nu = 2000
    for mu in [1.5, 2.5]:
        rng = np.random.default_rng(23)
        g = rng.standard_normal(d); g -= (g@bdir)*bdir; g /= np.linalg.norm(g); a = 0.7*bdir+np.sqrt(0.51)*g
        sfun = lambda X: 0.5+0.5*np.maximum(0.0, X@bdir)
        names = ["exact weighted (known w)","full-d estimated w","sketched k=1 known w_Z","sketched k=1 estimated w_Z","sketched k=8 estimated w_Z"]
        res = {nm:[] for nm in names}; errs = {nm:[] for nm in names}
        for r in range(200):
            Xtr = rng.standard_normal((n,d)); Ytr = Xtr@beta + sfun(Xtr)*rng.standard_normal(n)
            Xcal = rng.standard_normal((m,d)); Ycal = Xcal@beta + sfun(Xcal)*rng.standard_normal(m)
            Xte = mu*a + rng.standard_normal((ntest,d)); Yte = Xte@beta + sfun(Xte)*rng.standard_normal(ntest)
            Xu_s = rng.standard_normal((nu,d)); Xu_t = mu*a + rng.standard_normal((nu,d))
            bh,*_ = np.linalg.lstsq(np.c_[Xtr,np.ones(n)], Ytr, rcond=None); bh = bh[:d]/np.linalg.norm(bh[:d])
            Pi1 = bh[None,:]; Pi8 = orth_rows(np.vstack([bh, rng.standard_normal((7,d))]))
            def est_w(Pi):
                U = np.vstack([Xu_s@Pi.T, Xu_t@Pi.T]); lab = np.r_[np.zeros(nu), np.ones(nu)]
                clf = LogisticRegression(C=1e4, max_iter=2000).fit(U, lab)
                bhat_ = clf.coef_.ravel()
                return (lambda z: np.exp(clf.decision_function(z))), bhat_   # odds = w up to constant (n_t=n_s)
            def exact_err(bhat_, Pi):   # 1/2 E_{P_Z}|wbar - w_Z| = TV(N(bhat,I), N(mu Pi a, I)), exact under P_Z = N(0,I_k)
                return 2*norm.cdf(np.linalg.norm(bhat_ - mu*(Pi@a))/2) - 1
            wf = wexp(a,mu)
            res["exact weighted (known w)"].append(run(np.eye(d), wf, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            we, b_full = est_w(np.eye(d)); res["full-d estimated w"].append(run(np.eye(d), we, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            errs["full-d estimated w"].append(exact_err(b_full, np.eye(d)))
            res["sketched k=1 known w_Z"].append(run(Pi1, wexp(Pi1@a,mu), Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            we1, b1 = est_w(Pi1); res["sketched k=1 estimated w_Z"].append(run(Pi1, we1, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            errs["sketched k=1 estimated w_Z"].append(exact_err(b1, Pi1))
            we8, b8 = est_w(Pi8); res["sketched k=8 estimated w_Z"].append(run(Pi8, we8, Xtr,Ytr,Xcal,Ycal,Xte,Yte))
            errs["sketched k=8 estimated w_Z"].append(exact_err(b8, Pi8))
        print(f" mu={mu}: chi2_X={np.exp(mu**2)-1:.1f}")
        for nm,v in res.items():
            ex = f" exact 1/2E|wbar-w| = {np.mean(errs[nm]):.4f} (sd {np.std(errs[nm]):.4f})" if errs[nm] else ""
            report(nm, v, ex)

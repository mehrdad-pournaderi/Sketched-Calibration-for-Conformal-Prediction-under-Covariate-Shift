"""Shared helpers for the sketched-calibration experiments (E6-E8)."""
import numpy as np

alpha = 0.1

def orth_rows(M):
    q, _ = np.linalg.qr(M.T); return q.T

def wq_threshold(scores, w_cal, w_test, level):
    """Weighted split-conformal threshold (Tibshirani et al. 2019) with the test atom at +inf, vectorised over test weights."""
    order = np.argsort(scores); s, w = scores[order], w_cal[order]
    cw = np.cumsum(w); tot = cw[-1]
    idx = np.searchsorted(cw, level * (tot + w_test))
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)

def evaluate(scores_cal, w_cal, scores_te, w_te):
    """Coverage, median length proxy (2*threshold), fraction of infinite sets for absolute-residual scores."""
    q = wq_threshold(scores_cal, w_cal, w_te, 1 - alpha)
    return np.mean(scores_te <= q), np.median(2 * q), np.mean(np.isinf(q))

def sir_directions(X, r, k=1, H=10):
    """Sliced inverse regression of the (one-dimensional) response r on X; returns k orthonormal directions (rows)."""
    mu = X.mean(0); Xc = X - mu
    cov = np.cov(Xc, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    L = np.linalg.cholesky(cov); Linv = np.linalg.inv(L)
    W = Xc @ Linv.T                                   # whitened
    edges = np.quantile(r, np.linspace(0, 1, H + 1)); edges[-1] += 1e-9
    M = np.zeros((X.shape[1], X.shape[1]))
    for h in range(H):
        sel = (r >= edges[h]) & (r < edges[h + 1])
        if sel.sum() == 0: continue
        m = W[sel].mean(0); M += sel.mean() * np.outer(m, m)
    vals, vecs = np.linalg.eigh(M)
    V = vecs[:, ::-1][:, :k]                          # top-k eigenvectors in whitened coordinates
    D = (Linv.T @ V).T                                # back to x-coordinates: directions beta = Sigma^{-1/2} v
    return orth_rows(D)

def report(name, v, extra=""):
    v = np.array(v)
    return (f"  {name:40s} cov {v[:,0].mean():.3f}  sd {v[:,0].std():.3f}  P(cov<.85) {np.mean(v[:,0]<.85):.3f}"
            f"  len {np.median(v[:,1]):6.2f}  P(inf) {v[:,2].mean():.3f} {extra}")

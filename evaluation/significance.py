# -*- coding: utf-8 -*-
"""Per-endpoint significance of multi-visit gain (M2 vs M1).
Seed-ensembles each model's test predictions, then:
  - AUROC for M1 and M2
  - dAUROC = AUROC(M2) - AUROC(M1), with bootstrap 95% CI (paired, resample patients)
  - DeLong paired test p-value (correlated ROC AUCs)
Reads mvt_runs/{ep}_{H}yr/{ep}_{H}y_{mode}_s*/test_preds.csv  (pid,y,prob).
"""
import glob, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RUNS = "mvt_runs"; HOR = 1
EPS = ["HTN", "T2DM_lab", "Dyslipidemia", "MetS_4comp", "CAS_plaque"]

def seed_ens(ep, mode):
    fs = glob.glob(f"{RUNS}/{ep}_{HOR}yr/{ep}_{HOR}y_{mode}_s*/test_preds.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f)[["pid","y","prob"]] for f in fs])
    return d.groupby("pid", as_index=False).agg(y=("y","first"), prob=("prob","mean"))

# ---- fast DeLong (Sun & Xu 2014) for two paired predictors ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong_p(y, p1, p2):
    o = np.argsort(-y); y = y[o]; p1 = p1[o]; p2 = p2[o]
    m = int(y.sum()); n = len(y) - m
    preds = np.vstack([p1, p2])
    tx = np.vstack([_midrank(preds[r,:m]) for r in range(2)])
    ty = np.vstack([_midrank(preds[r,m:]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r,:])  for r in range(2)])
    aucs = tz[:,:m].sum(1)/m/n - (m+1)/2.0/n
    v01 = (tz[:,:m]-tx)/n; v10 = 1.0 - (tz[:,m:]-ty)/m
    cov = np.cov(v01)/m + np.cov(v10)/n
    L = np.array([1.0,-1.0]); var = float(L @ cov @ L)
    z = (aucs[0]-aucs[1]) / np.sqrt(var + 1e-12)
    return float(aucs[0]), float(aucs[1]), float(2*stats.norm.sf(abs(z)))

def boot_ci(y, p1, p2, B=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(B):
        idx = rng.randint(0, n, n); yy = y[idx]
        s = yy.sum()
        if s == 0 or s == n: continue
        d.append(roc_auc_score(yy, p2[idx]) - roc_auc_score(yy, p1[idx]))
    return np.percentile(d, [2.5, 97.5])

rows = []
for ep in EPS:
    m1, m2 = seed_ens(ep, "M1"), seed_ens(ep, "M2")
    if m1 is None or m2 is None:
        print("skip (missing preds):", ep); continue
    j = m1.merge(m2[["pid","prob"]].rename(columns={"prob":"prob2"}), on="pid")
    y = j["y"].values.astype(float); p1 = j["prob"].values; p2 = j["prob2"].values
    a1, a2, p = delong_p(y, p1, p2)
    lo, hi = boot_ci(y, p1, p2)
    rows.append({"endpoint": ep, "n": len(j), "pos": int(y.sum()),
                 "AUROC_M1": round(a1,4), "AUROC_M2": round(a2,4),
                 "dAUROC": round(a2-a1,4), "dAUROC_95CI": f"[{lo:+.4f},{hi:+.4f}]",
                 "DeLong_p": f"{p:.2e}", "sig": "***" if p<1e-3 else ("**" if p<1e-2 else ("*" if p<0.05 else "ns"))})

res = pd.DataFrame(rows)
print("\n=== Multi-visit (M2) vs single-visit (M1), seed-ensembled, test set ===")
print(res.to_string(index=False))
res.to_csv(f"{RUNS}/significance_M2_vs_M1.csv", index=False)
print(f"\nwrote {RUNS}/significance_M2_vs_M1.csv")

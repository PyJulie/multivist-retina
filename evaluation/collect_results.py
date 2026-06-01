# -*- coding: utf-8 -*-
"""Collect mvt_runs/*/*/summary.json -> tables. Prints first (csv save is best-effort)."""
import glob, json
import pandas as pd

rows = []
for f in glob.glob("mvt_runs/*/*/summary.json"):
    try:
        d = json.load(open(f)); a = d.get("args", {}) or {}
        rows.append({"endpoint": d.get("endpoint"), "mode": d.get("mode"),
                     "seed": d.get("seed"), "K_input": d.get("K_input"),
                     "exp_name": a.get("exp_name", ""), "test_auc": d.get("test_auc"),
                     "test_ap": d.get("test_ap"), "test_n": d.get("test_n"),
                     "test_n_pos": d.get("test_n_pos")})
    except Exception as e:
        print("skip", f, e)
if not rows:
    print("no summaries found"); raise SystemExit
df = pd.DataFrame(rows)
df["is_ksweep"] = df["exp_name"].astype(str).str.contains("_K")

# ---- main modes (exclude K-sweep) ----
main = df[~df["is_ksweep"]]
agg = (main.groupby(["endpoint","mode"])
          .agg(auc_mean=("test_auc","mean"), auc_std=("test_auc","std"),
               n_seeds=("test_auc","size"), test_n=("test_n","first"),
               test_pos=("test_n_pos","first")).reset_index())
print("\n=== modes x endpoint (mean over seeds) ===")
print(agg.to_string(index=False))

piv = agg.pivot(index="endpoint", columns="mode", values="auc_mean")
order = [c for c in ["LP1","LP2","M1","M2","M2nt","M3"] if c in piv.columns]
print("\n=== AUROC by mode (mean over seeds) ===")
print(piv[order].round(4).to_string())

if "M1" in piv.columns and "M2" in piv.columns:
    g = pd.DataFrame({"M1": piv["M1"], "M2": piv["M2"], "gain_M2_M1": piv["M2"]-piv["M1"]})
    if "M3" in piv.columns:
        g["M3_shuffle"] = piv["M3"]; g["gain_M3_M1"] = piv["M3"]-piv["M1"]
    if "LP1" in piv.columns and "LP2" in piv.columns:
        g["LP_gain"] = piv["LP2"]-piv["LP1"]
    print("\n=== multi-visit gain (M2-M1) vs time-shuffle control (M3-M1) ===")
    print(g.round(4).to_string())

# ---- K-sweep ----
ks = df[df["is_ksweep"]]
if len(ks):
    kp = (ks.groupby(["endpoint","K_input"]).agg(auc=("test_auc","mean")).reset_index()
            .pivot(index="endpoint", columns="K_input", values="auc"))
    print("\n=== K-sweep: M2 AUROC by K_input (mean over seeds) ===")
    print(kp.round(4).to_string())

# ---- best-effort save ----
for path in ["mvt_runs/all_results.csv", "multivist_ext/all_results.csv"]:
    try:
        df.drop(columns=["is_ksweep"]).to_csv(path, index=False)
        print("\nwrote", path); break
    except Exception as e:
        print("\n(could not write %s: %s)" % (path, e))

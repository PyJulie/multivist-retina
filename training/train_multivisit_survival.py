"""Discrete-time survival training for the multi-visit MACE model.

Replaces the binary classification head with a discrete-time hazard head
that outputs per-month probabilities over a [1, T_MAX] horizon. Loss is
the standard discrete-time NLL with right-censoring.

Inference produces cumulative incidence at any t in [1, T_MAX], so a single
trained model gives 1/3/5-year risk in one shot.

Built on top of train_multivisit.py: reuses PatientDataset (extended) and
MultiVisitTransformer (with replaced head).

Outputs same `summary.json`, `best.pt`, and `test_preds.csv` (with prob
at t=horizon) as train_multivisit.py for downstream compatibility.

Usage:
  python train_multivisit_survival.py --endpoint mace --horizon 5 \
                                       --T_MAX 60 --bin_months 1
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_multivisit import (
    PatientDataset, MultiVisitTransformer, collate, D_FEAT, SAMPLES_ROOT,
    fourier_time, RUNS_DIR,
)

try:
    from lifelines.utils import concordance_index
except ImportError:
    print("[warn] lifelines not available; C-index will be NaN.")
    concordance_index = None


# =====================================================================
# Survival dataset wrapper
# =====================================================================

class SurvivalPatientDataset(PatientDataset):
    """Same as PatientDataset but yields (t_event_months, e_event) instead
    of just `label`.

    t_event_months: integer month index (1..T_MAX) of event or censoring
    e_event:        1 if event observed, 0 if censored
    """
    def __init__(self, *args, T_MAX: int = 60, bin_months: int = 1,
                 endpoint: str = "mace", **kwargs):
        super().__init__(*args, **kwargs)
        self.T_MAX = T_MAX
        self.bin_months = bin_months
        self.endpoint = endpoint

    def __getitem__(self, i):
        out = super().__getitem__(i)
        s = self.df.iloc[i]
        # Event time:
        #   if label==1 → use earliest_<ep> - index_date (in days) /30.4
        #   else        → use follow_up_days /30.4 as censoring time
        # Capped at T_MAX.
        label = int(s["label"])
        if label == 1:
            ev_date = s.get(f"earliest_{self.endpoint}",
                            s.get("earliest_mace"))
            idx_date = s["index_date"]
            try:
                days = float((ev_date - idx_date).days)
            except Exception:
                days = float(s.get("days_to_event", 0))
        else:
            days = float(s["follow_up_days"])
        days = max(days, 1.0)
        months_raw = days / 30.4375
        # bin index in [1, T_MAX] (clamp top)
        t_bin = int(np.ceil(months_raw / self.bin_months))
        t_bin = max(1, min(t_bin, self.T_MAX // self.bin_months))
        # If the event happened *after* T_MAX it becomes effectively censored
        if label == 1 and t_bin > self.T_MAX // self.bin_months:
            label = 0
        out["t_bin"]   = torch.tensor(t_bin,  dtype=torch.long)
        out["e_evt"]   = torch.tensor(label,  dtype=torch.long)
        out["t_days"]  = torch.tensor(days,   dtype=torch.float32)
        return out


def collate_surv(batch):
    out = collate(batch)
    out["t_bin"]  = torch.stack([b["t_bin"]  for b in batch])
    out["e_evt"]  = torch.stack([b["e_evt"]  for b in batch])
    out["t_days"] = torch.stack([b["t_days"] for b in batch])
    return out


# =====================================================================
# Survival head & loss
# =====================================================================

class SurvivalMultiVisitTransformer(nn.Module):
    """Wrap MultiVisitTransformer's encoder; replace its scalar head with
    a discrete-time hazard head outputting (B, n_bins) logits per bin.
    """
    def __init__(self, n_bins: int, d_in: int = D_FEAT, d: int = 256,
                 n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, no_time: bool = False):
        super().__init__()
        # We piggy-back on the existing module structure to keep checkpoint
        # compatibility friendly.
        self.proj      = nn.Linear(d_in, d)
        self.eye_emb   = nn.Embedding(2, d)
        self.no_time   = no_time
        self.time_proj = None if no_time else nn.Linear(16, d)
        self.cls       = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder   = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Head outputs n_bins logits → sigmoid → per-bin discrete-time
        # hazard h_t = P(event in bin t | survived to bin t).
        # Survival to time t: S_t = prod_{k<=t} (1 - h_k).
        # P(event in bin t): f_t = h_t * S_{t-1}.
        # This avoids the softmax-induced S(T_MAX)=0 trap.
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, n_bins))
        self.n_bins = n_bins
        # Initialize hazard bias so initial per-bin hazard ~ 0.5% (assuming
        # ~10% event rate over n_bins). This keeps initial S(T) ≈ 0.9 rather
        # than collapsing to 0 with sigmoid(0)=0.5.
        target_haz = 0.005
        bias_init = float(np.log(target_haz / (1 - target_haz)))   # ~ -5.3
        with torch.no_grad():
            self.head[-1].bias.fill_(bias_init)

    def forward(self, emb, eye, dt, mask):
        B = emb.size(0)
        h = self.proj(emb) + self.eye_emb(eye)
        if not self.no_time:
            h = h + self.time_proj(fourier_time(dt))
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_pad, mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=full_mask)
        cls_out = h[:, 0]                            # (B, d)
        logits = self.head(cls_out)                  # (B, n_bins)
        return logits


def discrete_nll(logits, t_bin, e_evt):
    """Discrete-time survival NLL via per-bin hazards.

    logits  (B, n_bins) — pre-sigmoid hazard logits
    t_bin   (B,)        — bin index in [1, n_bins] (event time or censor)
    e_evt   (B,)        — 1 = event observed at t_bin, 0 = censored at t_bin

    Define h_t = sigmoid(logits[:, t-1]) = P(event in bin t | alive at t-1).
    Then:
      S_t        = prod_{k<=t} (1 - h_k)              (survival to end of t)
      f_t        = h_t * S_{t-1}                       (PMF at bin t)
      log f_t    = log h_t + sum_{k<t} log(1 - h_k)
      log S_t    = sum_{k<=t} log(1 - h_k)

    NLL contribution:
      event    : -log f_{t_bin}   = -[log h_{t_bin} + sum_{k<t_bin} log(1-h_k)]
      censored : -log S_{t_bin}   = -sum_{k<=t_bin} log(1 - h_k)
    """
    eps = 1e-7
    h = torch.sigmoid(logits)                              # (B, n_bins)
    log_h         = torch.log(h.clamp(min=eps))            # (B, n_bins)
    log_one_min_h = torch.log((1.0 - h).clamp(min=eps))    # (B, n_bins)

    B, n_bins = logits.shape
    idx = (t_bin - 1).clamp(0, n_bins - 1)                 # 0-indexed bin

    # Cumulative log-survival up to and including bin k (for each k).
    cum_log_one_min_h = log_one_min_h.cumsum(dim=1)        # (B, n_bins)
    # log S_{t_bin} = cum_log_one_min_h[:, idx]
    log_S_t = cum_log_one_min_h.gather(1, idx.unsqueeze(1)).squeeze(1)
    # log S_{t_bin - 1} = cum_log_one_min_h[:, idx-1]; if idx==0 → 0
    prev_idx = (idx - 1).clamp(min=0)
    log_S_prev = cum_log_one_min_h.gather(1, prev_idx.unsqueeze(1)).squeeze(1)
    log_S_prev = torch.where(idx == 0,
                              torch.zeros_like(log_S_prev),
                              log_S_prev)
    log_h_t = log_h.gather(1, idx.unsqueeze(1)).squeeze(1)
    log_f_t = log_h_t + log_S_prev                          # log P(event at t_bin)

    e = e_evt.float()
    nll = - (e * log_f_t + (1.0 - e) * log_S_t)
    return nll.mean()


def cif_at(logits, t):
    """Cumulative incidence at bin t (1-indexed: events in bins 1..t).

    Returns (B,) with values in [0, 1].
    """
    eps = 1e-7
    h = torch.sigmoid(logits)
    # 1 - prod_{k<=t} (1-h_k) = 1 - exp(sum_{k<=t} log(1-h_k))
    log_one_min_h = torch.log((1.0 - h).clamp(min=eps))
    log_S_t = log_one_min_h[:, :t].sum(dim=1)
    return 1.0 - torch.exp(log_S_t)


# =====================================================================
# Eval
# =====================================================================

@torch.no_grad()
def evaluate_surv(model, loader, device, eval_bins=(12, 36, 60),
                  bin_months: int = 1):
    model.eval()
    all_logits, all_t, all_e, all_pid = [], [], [], []
    for batch in loader:
        emb  = batch["emb"].to(device)
        eye  = batch["eye"].to(device)
        dt   = batch["dt"].to(device)
        mask = batch["mask"].to(device)
        logits = model(emb, eye, dt, mask)
        all_logits.append(logits.cpu())
        all_t.append(batch["t_bin"])
        all_e.append(batch["e_evt"])
        all_pid.extend(batch["pids"])
    logits = torch.cat(all_logits)
    t = torch.cat(all_t).numpy()
    e = torch.cat(all_e).numpy()

    out = {"n": len(t), "n_event": int(e.sum())}

    # Risk score = CIF at max bin → for global C-index
    risk_max = cif_at(logits, logits.shape[1]).numpy()
    if concordance_index is not None:
        try:
            cidx = concordance_index(t, -risk_max, e)
            out["c_index"] = float(cidx)
        except Exception as ex:
            out["c_index"] = float("nan")
            print(f"  [warn] c-index failed: {ex}")
    else:
        out["c_index"] = float("nan")

    # Fixed-horizon AUROC: at month H, treat patients with t<=H & e==1 as
    # positive; t>=H regardless of event as negatives. Patients censored
    # before H are dropped (standard fixed-horizon AUROC).
    for H in eval_bins:
        H_bin = H // bin_months
        risk_H = cif_at(logits, H_bin).numpy()
        eligible = (t >= H_bin) | (e == 1)
        # Positives are those with event by H
        y = np.zeros_like(e)
        y[(t <= H_bin) & (e == 1)] = 1
        m = eligible
        try:
            auc = roc_auc_score(y[m], risk_H[m])
            out[f"auc_t{H}"] = float(auc)
            out[f"n_t{H}"] = int(m.sum())
            out[f"n_pos_t{H}"] = int(y[m].sum())
        except Exception:
            out[f"auc_t{H}"] = float("nan")

    return out, all_pid, risk_max


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="mace")
    ap.add_argument("--horizon", type=int, default=5,
                    help="primary reporting horizon (years)")
    ap.add_argument("--K_max", type=int, default=8)
    ap.add_argument("--T_MAX", type=int, default=60,
                    help="maximum followup months tracked by model")
    ap.add_argument("--bin_months", type=int, default=1)
    ap.add_argument("--samples_root", default=SAMPLES_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--no_time", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_bins = args.T_MAX // args.bin_months
    name = args.exp_name or \
           f"M2surv_{args.endpoint}{args.horizon}_T{args.T_MAX}_s{args.seed}"
    out_dir = Path(RUNS_DIR) / f"{args.endpoint}{args.horizon}yr" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = Path(args.samples_root) / f"{args.endpoint}{args.horizon}yr"
    print(f"  endpoint={args.endpoint}  horizon={args.horizon}  "
          f"T_MAX={args.T_MAX}m bin={args.bin_months}m n_bins={n_bins}")

    ds_kw = dict(K_input=args.K_max, T_MAX=args.T_MAX,
                  bin_months=args.bin_months, endpoint=args.endpoint)
    ds_tr = SurvivalPatientDataset(str(samples / "train.pkl"),
                                    train_random_K=True, seed=args.seed,
                                    **ds_kw)
    ds_va = SurvivalPatientDataset(str(samples / "val.pkl"),
                                    seed=args.seed, **ds_kw)
    ds_te = SurvivalPatientDataset(str(samples / "test.pkl"),
                                    seed=args.seed, **ds_kw)
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, collate_fn=collate_surv)
    dl_tr = DataLoader(ds_tr, shuffle=True, **dl_kw)
    dl_va = DataLoader(ds_va, shuffle=False, **dl_kw)
    dl_te = DataLoader(ds_te, shuffle=False, **dl_kw)
    print(f"  splits: tr={len(ds_tr)} va={len(ds_va)} te={len(ds_te)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurvivalMultiVisitTransformer(n_bins=n_bins, no_time=args.no_time
                                          ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = -1.0; best_ep = -1; bad = 0
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0; n_seen = 0
        for batch in dl_tr:
            emb  = batch["emb"].to(device)
            eye  = batch["eye"].to(device)
            dt   = batch["dt"].to(device)
            mask = batch["mask"].to(device)
            t_bin = batch["t_bin"].to(device)
            e_evt = batch["e_evt"].to(device)
            logits = model(emb, eye, dt, mask)
            loss = discrete_nll(logits, t_bin, e_evt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * emb.size(0)
            n_seen += emb.size(0)
        sched.step()

        # Val
        val_out, _, _ = evaluate_surv(model, dl_va, device,
                                       eval_bins=(12, 36, 60),
                                       bin_months=args.bin_months)
        val_metric = val_out.get("c_index", val_out.get("auc_t60", 0))
        history.append({"epoch": ep, "train_loss": loss_sum / n_seen,
                        **val_out})
        print(f"  ep {ep:3d}  loss={loss_sum/n_seen:.4f}  "
              f"val_C={val_out.get('c_index', np.nan):.4f}  "
              f"auc(1y)={val_out.get('auc_t12', np.nan):.4f}  "
              f"auc(3y)={val_out.get('auc_t36', np.nan):.4f}  "
              f"auc(5y)={val_out.get('auc_t60', np.nan):.4f}")
        if val_metric > best_val:
            best_val = val_metric; best_ep = ep; bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop @ ep {ep}")
                break

    # Test
    print(f"\n  loading best.pt (val C-index={best_val:.4f} @ep {best_ep})")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_out, pids, risk_max = evaluate_surv(model, dl_te, device,
                                              eval_bins=(12, 36, 60),
                                              bin_months=args.bin_months)
    print("  test:", test_out)

    # Save preds (for compat with binary pipelines, save risk at primary horizon)
    H_primary = args.horizon * 12
    H_bin = H_primary // args.bin_months
    # Recompute test risk at H_primary
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dl_te:
            emb  = batch["emb"].to(device)
            eye  = batch["eye"].to(device)
            dt   = batch["dt"].to(device)
            mask = batch["mask"].to(device)
            logits = model(emb, eye, dt, mask)
            cif = cif_at(logits, H_bin).cpu().numpy()
            for pid, p, y, t, e in zip(batch["pids"], cif,
                                        batch.get("y", batch["e_evt"]).numpy(),
                                        batch["t_bin"].numpy(),
                                        batch["e_evt"].numpy()):
                rows.append({"pid": pid, "y": int((t <= H_bin) and (e == 1)),
                             "prob": float(p), "t_bin": int(t),
                             "e_evt": int(e)})
    pd.DataFrame(rows).to_csv(out_dir / "test_preds.csv", index=False)

    summary = {
        "mode": "M2_surv",
        "exp_name": name,
        "endpoint": args.endpoint,
        "horizon": args.horizon,
        "T_MAX": args.T_MAX, "bin_months": args.bin_months,
        "n_bins": n_bins, "K_max": args.K_max, "seed": args.seed,
        "best_val_metric": float(best_val),
        "best_val_epoch": int(best_ep),
        "test": test_out,
        "elapsed_min": (time.time() - t0) / 60.0,
        "args": vars(args),
        "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[wrote] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

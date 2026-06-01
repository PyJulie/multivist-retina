"""Multi-task M2: shared encoder + per-endpoint heads.

Trains one MultiVisitTransformer encoder with 4 task heads (MACE, MI,
stroke, all-cause death) on the intersection of the 4 cohorts (~97% of
MACE cohort).

For each batch, the four heads produce four logits; loss is the sum of
per-endpoint BCE-with-pos-weight losses.

Usage:
  python train_multivisit_multitask.py --horizon 3 --seed 0 \\
         --K_max 8 --exp_name M2mt_h3_s0
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
    RUNS_DIR, fourier_time, get_emb,
)


ENDPOINTS_DEFAULT = ["mace", "mi", "stroke", "alldeath"]
# Allow override via CLI
ENDPOINTS = list(ENDPOINTS_DEFAULT)


def build_intersection_df(horizon: int, samples_root: str, split: str):
    """Load all 4 endpoint pkls, intersect by pid, return a DataFrame
    with all 4 binary labels per patient (pid, label_mace, label_mi, ...)
    along with input image fields from the MACE pkl as canonical."""
    # Use the first endpoint as the canonical "image source" cohort.
    canonical = ENDPOINTS[0]
    df_can = pickle.load(open(
        Path(samples_root) / f"{canonical}{horizon}yr" / f"{split}.pkl", "rb"))
    base = df_can.set_index("patient_id").copy()
    base[f"label_{canonical}"] = base["label"].astype(int)
    others = [e for e in ENDPOINTS if e != canonical]
    for ep in others:
        d = pickle.load(open(
            Path(samples_root) / f"{ep}{horizon}yr" / f"{split}.pkl", "rb"))
        base[f"label_{ep}"] = d.set_index("patient_id")["label"].astype(int)
    # Keep only patients with labels in all chosen endpoints (intersection)
    base = base.dropna(subset=[f"label_{e}" for e in ENDPOINTS])
    for e in ENDPOINTS:
        base[f"label_{e}"] = base[f"label_{e}"].astype(int)
    return base.reset_index()


class MultiTaskDataset(Dataset):
    AGE_MEAN = 66.7
    AGE_STD  = 14.0

    def __init__(self, df, K_input: int, train_random_K: bool = False,
                  seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.K_input = K_input
        self.train_random_K = train_random_K
        self.rng = np.random.RandomState(seed)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        s = self.df.iloc[i]
        paths = s["input_paths"]; lats = s["input_lat"]
        dts = s["input_dt_days"]; dates = s["input_dates"]
        unique_dates = sorted(set(dates))
        if self.train_random_K:
            K = int(self.rng.randint(1, min(self.K_input,
                                              len(unique_dates)) + 1))
        else:
            K = min(self.K_input, len(unique_dates))
        keep_dates = set(sorted(unique_dates)[-K:])
        keep_idx = [j for j, d in enumerate(dates) if d in keep_dates]
        embs = np.stack([get_emb(paths[j]) for j in keep_idx])
        eye  = np.array([0 if str(lats[j]).upper().startswith("R") else 1
                          for j in keep_idx], dtype=np.int64)
        dt = np.array([dts[j] for j in keep_idx], dtype=np.float32)
        out = {
            "emb":   torch.from_numpy(embs),
            "eye":   torch.from_numpy(eye),
            "dt":    torch.from_numpy(dt),
            "T":     len(keep_idx),
            "pid":   s["patient_id"],
        }
        for ep in ENDPOINTS:
            out[f"label_{ep}"] = torch.tensor(int(s[f"label_{ep}"]),
                                                dtype=torch.float32)
        return out


def collate_mt(batch):
    Tmax = max(b["T"] for b in batch)
    B = len(batch)
    emb  = torch.zeros(B, Tmax, D_FEAT)
    eye  = torch.zeros(B, Tmax, dtype=torch.long)
    dt   = torch.zeros(B, Tmax)
    mask = torch.ones(B, Tmax, dtype=torch.bool)
    pids = []
    for i, b in enumerate(batch):
        T = b["T"]
        emb[i, :T] = b["emb"]; eye[i, :T] = b["eye"]; dt[i, :T] = b["dt"]
        mask[i, :T] = False
        pids.append(b["pid"])
    out = {"emb": emb, "eye": eye, "dt": dt, "mask": mask, "pids": pids}
    for ep in ENDPOINTS:
        out[f"label_{ep}"] = torch.stack([b[f"label_{ep}"] for b in batch])
    return out


class MultiTaskTransformer(nn.Module):
    """Shared encoder + 4 task-specific heads."""
    def __init__(self, d_in=D_FEAT, d=256, n_layers=2, n_heads=4,
                  dropout=0.1):
        super().__init__()
        self.proj      = nn.Linear(d_in, d)
        self.eye_emb   = nn.Embedding(2, d)
        self.time_proj = nn.Linear(16, d)
        self.cls       = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d,
                                            dropout=dropout, batch_first=True,
                                            activation="gelu", norm_first=True)
        self.encoder   = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleDict({
            ep: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            for ep in ENDPOINTS
        })

    def forward(self, emb, eye, dt, mask):
        B = emb.size(0)
        h = self.proj(emb) + self.eye_emb(eye)
        h = h + self.time_proj(fourier_time(dt))
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_pad, mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=full_mask)
        cls_out = h[:, 0]
        return {ep: self.heads[ep](cls_out).squeeze(-1) for ep in ENDPOINTS}


@torch.no_grad()
def evaluate_mt(model, loader, device):
    model.eval()
    by_ep = {ep: {"y": [], "p": []} for ep in ENDPOINTS}
    for batch in loader:
        emb  = batch["emb"].to(device); eye = batch["eye"].to(device)
        dt   = batch["dt"].to(device);  mask = batch["mask"].to(device)
        out = model(emb, eye, dt, mask)
        for ep in ENDPOINTS:
            by_ep[ep]["y"].extend(batch[f"label_{ep}"].numpy().tolist())
            by_ep[ep]["p"].extend(torch.sigmoid(out[ep]).cpu().numpy().tolist())
    metrics = {}
    for ep in ENDPOINTS:
        y = np.asarray(by_ep[ep]["y"]); p = np.asarray(by_ep[ep]["p"])
        if len(set(y)) > 1:
            metrics[f"auc_{ep}"] = float(roc_auc_score(y, p))
            metrics[f"ap_{ep}"]  = float(average_precision_score(y, p))
            metrics[f"brier_{ep}"] = float(brier_score_loss(y, p))
            metrics[f"n_pos_{ep}"] = int(y.sum())
        metrics[f"n_{ep}"] = int(len(y))
    return metrics, by_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--K_max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--samples_root", default=SAMPLES_ROOT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--task_weights", nargs="+", type=float, default=None,
                    help="weights for the chosen endpoints' losses (in order)")
    ap.add_argument("--endpoints", nargs="+",
                    default=ENDPOINTS_DEFAULT,
                    help="subset of endpoints to multi-task on")
    args = ap.parse_args()

    # Mutate global ENDPOINTS so downstream functions see the chosen subset
    global ENDPOINTS
    ENDPOINTS[:] = args.endpoints

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    weights = args.task_weights or [1.0] * len(ENDPOINTS)
    assert len(weights) == len(ENDPOINTS)
    weights = dict(zip(ENDPOINTS, weights))

    out_dir = Path(RUNS_DIR) / f"multitask_h{args.horizon}" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"=== {args.exp_name} multitask seed={args.seed} ===")
    log(f"  args: {vars(args)}")

    df_tr = build_intersection_df(args.horizon, args.samples_root, "train")
    df_va = build_intersection_df(args.horizon, args.samples_root, "val")
    df_te = build_intersection_df(args.horizon, args.samples_root, "test")
    log(f"  intersected splits: tr={len(df_tr)} va={len(df_va)} te={len(df_te)}")
    for ep in ENDPOINTS:
        log(f"    {ep}: train pos={int(df_tr[f'label_{ep}'].sum())} "
             f"({df_tr[f'label_{ep}'].mean()*100:.1f}%)")

    ds_tr = MultiTaskDataset(df_tr, args.K_max, train_random_K=True,
                              seed=args.seed)
    ds_va = MultiTaskDataset(df_va, args.K_max, train_random_K=False,
                              seed=args.seed)
    ds_te = MultiTaskDataset(df_te, args.K_max, train_random_K=False,
                              seed=args.seed)
    dl_kw = dict(num_workers=args.num_workers, pin_memory=True,
                  collate_fn=collate_mt)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, **dl_kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, **dl_kw)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, **dl_kw)

    pos_weights = {}
    for ep in ENDPOINTS:
        pos = int(df_tr[f"label_{ep}"].sum())
        neg = len(df_tr) - pos
        pos_weights[ep] = float(neg) / max(pos, 1)
        log(f"  pos_weight[{ep}]={pos_weights[ep]:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTransformer().to(device)
    log(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    losses = {ep: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weights[ep]], device=device))
              for ep in ENDPOINTS}

    history = []
    best_score = -1.0; best_ep = -1; bad = 0; t0 = time.time()
    for ep_idx in range(1, args.epochs + 1):
        model.train()
        sum_total, sum_per = 0.0, {ep: 0.0 for ep in ENDPOINTS}
        n = 0
        for batch in dl_tr:
            emb  = batch["emb"].to(device); eye = batch["eye"].to(device)
            dt   = batch["dt"].to(device); mask = batch["mask"].to(device)
            out  = model(emb, eye, dt, mask)
            losses_per = {}
            for ep in ENDPOINTS:
                y = batch[f"label_{ep}"].to(device)
                losses_per[ep] = losses[ep](out[ep], y)
            total = sum(weights[ep] * losses_per[ep] for ep in ENDPOINTS)
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            B = emb.size(0)
            sum_total += float(total.item()) * B; n += B
            for ep in ENDPOINTS:
                sum_per[ep] += float(losses_per[ep].item()) * B
        sched.step()

        m, _ = evaluate_mt(model, dl_va, device)
        # Use mean val AUC across 4 endpoints as monitoring score
        avg_auc = np.mean([m[f"auc_{ep}"] for ep in ENDPOINTS])
        history.append({"epoch": ep_idx, "loss": sum_total / n,
                         **{f"loss_{ep}": sum_per[ep] / n for ep in ENDPOINTS},
                         **m, "avg_val_auc": float(avg_auc)})
        log(f"  ep {ep_idx:3d}  loss={sum_total/n:.4f}  "
             + " ".join([f"auc_{ep}={m[f'auc_{ep}']:.4f}"
                          for ep in ENDPOINTS])
             + f"  avg={avg_auc:.4f}")
        if avg_auc > best_score:
            best_score = avg_auc; best_ep = ep_idx; bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  early stop @ ep {ep_idx}")
                break

    # Test
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_m, test_ep_data = evaluate_mt(model, dl_te, device)
    log("\n  test:")
    for ep in ENDPOINTS:
        log(f"    {ep}: auc={test_m[f'auc_{ep}']:.4f}  "
             f"ap={test_m[f'ap_{ep}']:.4f}  "
             f"brier={test_m[f'brier_{ep}']:.4f}  "
             f"n={test_m[f'n_{ep}']}")

    # Save per-endpoint preds
    for ep in ENDPOINTS:
        preds = pd.DataFrame({
            "pid": [b for batch in DataLoader(ds_te, batch_size=512,
                     collate_fn=collate_mt) for b in batch["pids"]],
            "y":   test_ep_data[ep]["y"],
            "prob": test_ep_data[ep]["p"],
        })
        preds.to_csv(out_dir / f"test_preds_{ep}.csv", index=False)

    summary = {"mode": "M2mt", "exp_name": args.exp_name,
                "horizon": args.horizon, "seed": args.seed,
                "K_max": args.K_max,
                "best_val_avg_auc": float(best_score),
                "best_val_epoch": int(best_ep),
                "test_metrics": test_m,
                "elapsed_min": (time.time() - t0) / 60.0,
                "args": vars(args),
                "history": history}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"\n  wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

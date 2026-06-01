"""Multi-visit MACE prediction — unified train/eval for 5 modes:

  LP1  : Linear probe on mean(OD,OS) of index visit only
  LP2  : Linear probe on mean over all tokens (all visits, all eyes)
  M1   : Transformer aggregator, K=1 (index visit only, 2 tokens)
  M2   : Transformer aggregator, K=K_max (multi-visit)
  M3   : M2 with Δt shuffled (control for time signal)

All modes share:
  * RetFound frozen features (read from cache)
  * Same patient samples (mace3yr/{train,val,test}.pkl)
  * Same train/val/test split
  * Same loss (BCE w/ pos_weight), same evaluation

Usage:
  python train_multivisit.py --mode M2 --seed 0 --exp_name M2_s0
  python train_multivisit.py --mode M1 --seed 0 --exp_name M1_s0
  ...
"""
import argparse
import hashlib
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from torch.utils.data import DataLoader, Dataset


# ============================ paths ============================
SAMPLES_ROOT = "${SAMPLES_DIR}"
CACHE_DIR    = "${CACHE_DIR}"
# Allow override via env var OCULOMICS_PACK_DIRS=path1:path2:...
_DEFAULT_PACK_DIRS = [
    "${CACHE_DIR}",
    "${CACHE_DIR}_extra",
    "${CACHE_DIR}_K16",
    "${CACHE_DIR}_K24",
]
_env_packs = os.environ.get("OCULOMICS_PACK_DIRS")
PACK_DIRS = _env_packs.split(":") if _env_packs else _DEFAULT_PACK_DIRS
RUNS_DIR     = "${RUNS_DIR}"

D_FEAT = 1024   # RetFound CLS dim


def path_md5(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()


def cached(p: str) -> str:
    h = path_md5(p)
    return os.path.join(CACHE_DIR, h[:2], f"{h}.npy")


# Memmap loader: load once at module import (or on first call).
# Each pack dir contains embeddings.bin + index.json + meta.json.
_PACK_LOADED = False
_PACK = []   # list of (mm, index_dict)

def _load_packs():
    global _PACK_LOADED, _PACK
    if _PACK_LOADED:
        return
    for pd in PACK_DIRS:
        if not os.path.isdir(pd):
            continue
        meta = json.load(open(os.path.join(pd, "meta.json")))
        idx  = json.load(open(os.path.join(pd, "index.json")))
        mm = np.memmap(os.path.join(pd, "embeddings.bin"),
                       dtype=meta["dtype"], mode="r",
                       shape=(meta["N"], meta["D"]))
        _PACK.append((mm, idx))
        print(f"  [pack] loaded {pd}  N={meta['N']:,}  D={meta['D']}")
    _PACK_LOADED = True


def get_emb(path: str) -> np.ndarray:
    """Look up embedding for an image path. Tries packed memmaps first,
    then falls back to per-file .npy in CACHE_DIR."""
    _load_packs()
    h = path_md5(path)
    for mm, idx in _PACK:
        i = idx.get(h)
        if i is not None:
            return np.asarray(mm[i], dtype=np.float32)
    # Fallback: per-file .npy (legacy)
    try:
        return np.load(cached(path)).astype(np.float32)
    except Exception:
        return np.zeros(D_FEAT, dtype=np.float32)


# ============================ dataset ============================
class PatientDataset(Dataset):
    """Per-patient sample with cached RetFound features.

    Modes:
      K_input = 1     : keep only rows at index_date (M1 / LP1)
      K_input = >1    : keep tail K visit-dates (M2 / LP2 / M3)
      shuffle_dt       : permute Δt within each sample (M3 ablation)
      train_random_K   : during training, randomly subsample K∈[1,K_max] visit-dates
                         (used for M2 only — exposes it to short sequences)
    """
    AGE_MEAN = 66.7      # rough train cohort mean
    AGE_STD  = 14.0

    def __init__(self, pkl_path: str, K_input: int,
                 shuffle_dt: bool = False, train_random_K: bool = False,
                 seed: int = 0, with_tabular: bool = False):
        self.df = pickle.load(open(pkl_path, "rb"))
        self.K_input = K_input
        self.shuffle_dt = shuffle_dt
        self.train_random_K = train_random_K
        self.with_tabular = with_tabular
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def _load_emb(self, p: str) -> np.ndarray:
        return get_emb(p)

    def __getitem__(self, i):
        s = self.df.iloc[i]
        paths = s["input_paths"]
        lats  = s["input_lat"]
        dts   = s["input_dt_days"]   # int days, ≥ 0
        dates = s["input_dates"]

        # Choose K visit-dates
        unique_dates = sorted(set(dates))
        if self.train_random_K:
            K = int(self.rng.randint(1, min(self.K_input, len(unique_dates)) + 1))
        else:
            K = min(self.K_input, len(unique_dates))
        keep_dates = set(sorted(unique_dates)[-K:])  # last K dates

        keep_idx = [j for j, d in enumerate(dates) if d in keep_dates]
        embs = np.stack([self._load_emb(paths[j]) for j in keep_idx])  # (T, 1024)
        eye  = np.array([0 if str(lats[j]).upper().startswith("R") else 1
                         for j in keep_idx], dtype=np.int64)
        dt   = np.array([dts[j] for j in keep_idx], dtype=np.float32)

        if self.shuffle_dt:
            self.rng.shuffle(dt)

        out = {
            "emb":   torch.from_numpy(embs),                  # (T, 1024)
            "eye":   torch.from_numpy(eye),                   # (T,)
            "dt":    torch.from_numpy(dt),                    # (T,)
            "label": torch.tensor(int(s["label"]), dtype=torch.float32),
            "pid":   s["patient_id"],
            "T":     len(keep_idx),
        }
        if self.with_tabular:
            age = float(s.get("age_at_index", np.nan))
            if not np.isfinite(age):
                age = self.AGE_MEAN
            age_n = (age - self.AGE_MEAN) / self.AGE_STD
            sex_str = str(s.get("sex", "")).upper()
            sex_v = 1.0 if sex_str.startswith("M") else 0.0
            out["tab"] = torch.tensor([age_n, sex_v], dtype=torch.float32)
        return out


def collate(batch):
    """Pad to max T; mask = True for padding."""
    Tmax = max(b["T"] for b in batch)
    B = len(batch)
    emb  = torch.zeros(B, Tmax, D_FEAT)
    eye  = torch.zeros(B, Tmax, dtype=torch.long)
    dt   = torch.zeros(B, Tmax)
    mask = torch.ones(B, Tmax, dtype=torch.bool)
    y    = torch.zeros(B)
    pids = []
    has_tab = "tab" in batch[0]
    tab = torch.zeros(B, 2) if has_tab else None
    for i, b in enumerate(batch):
        T = b["T"]
        emb[i, :T] = b["emb"]
        eye[i, :T] = b["eye"]
        dt [i, :T] = b["dt"]
        mask[i, :T] = False
        y[i] = b["label"]
        pids.append(b["pid"])
        if has_tab:
            tab[i] = b["tab"]
    out = {"emb": emb, "eye": eye, "dt": dt, "mask": mask,
           "y": y, "pids": pids}
    if has_tab:
        out["tab"] = tab
    return out


# ============================ models ============================
def fourier_time(dt_days: torch.Tensor, num_bands: int = 8) -> torch.Tensor:
    """(B, T) Δt days → (B, T, 2*num_bands) sin/cos features."""
    scales = torch.tensor([1, 7, 30, 90, 180, 365, 730, 1825],
                          device=dt_days.device, dtype=dt_days.dtype)[:num_bands]
    x = dt_days.unsqueeze(-1) / scales
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiVisitTransformer(nn.Module):
    """Transformer over per-image tokens with eye + Δt embeddings.

    For M1 (K=1, T=2 tokens) and M2 (K=K_max) — same architecture, only
    differ in input length at runtime.

    no_time=True: skip time embedding entirely (Set-Transformer-style,
    used for M2nt ablation testing whether Δt contributes anything).
    """
    def __init__(self, d_in: int = D_FEAT, d: int = 256,
                 n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, no_time: bool = False,
                 d_tab: int = 0):
        super().__init__()
        self.proj      = nn.Linear(d_in, d)
        self.eye_emb   = nn.Embedding(2, d)
        self.no_time   = no_time
        self.time_proj = None if no_time else nn.Linear(16, d)
        self.cls       = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4*d,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder   = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_tab = d_tab
        head_in = d + d_tab
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, 1))

    def forward(self, emb, eye, dt, mask, tab=None):
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
        if self.d_tab > 0:
            assert tab is not None, "model expects tab features"
            cls_out = torch.cat([cls_out, tab], dim=-1)
        return self.head(cls_out).squeeze(-1)


class LinearProbe(nn.Module):
    """Mean-pool features (over non-pad tokens) → Linear.

    For LP1 (K=1) and LP2 (K=K_max). Strict linear except for the mean-pool.
    No eye/time embedding.
    """
    def __init__(self, d_in: int = D_FEAT):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, emb, eye, dt, mask):
        # mean over non-pad tokens
        valid = (~mask).unsqueeze(-1).float()       # (B, T, 1)
        s = (emb * valid).sum(1)
        n = valid.sum(1).clamp(min=1.0)
        pooled = s / n                                # (B, D)
        return self.head(pooled).squeeze(-1)


# ============================ eval ============================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_y, all_pid = [], [], []
    for batch in loader:
        emb = batch["emb"].to(device, non_blocking=True)
        eye = batch["eye"].to(device, non_blocking=True)
        dt  = batch["dt"].to(device, non_blocking=True)
        mask= batch["mask"].to(device, non_blocking=True)
        logit = model(emb, eye, dt, mask).float()
        prob = torch.sigmoid(logit).cpu().numpy()
        all_p.extend(prob.tolist())
        all_y.extend(batch["y"].numpy().tolist())
        all_pid.extend(batch["pids"])
    df = pd.DataFrame({"pid": all_pid, "y": all_y, "prob": all_p})
    auc = roc_auc_score(df["y"], df["prob"]) if df["y"].nunique() > 1 else float("nan")
    ap  = average_precision_score(df["y"], df["prob"]) if df["y"].nunique() > 1 else float("nan")
    brier = brier_score_loss(df["y"], df["prob"])
    return {"auc": float(auc), "ap": float(ap), "brier": float(brier),
            "n": len(df), "n_pos": int(df["y"].sum()), "preds": df}


# ============================ train ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["LP1", "LP2", "M1", "M2", "M3", "M2nt",
                             "M2tab", "M1tab", "M2big"])
    ap.add_argument("--agg_d", type=int, default=256)
    ap.add_argument("--agg_layers", type=int, default=2)
    ap.add_argument("--agg_heads", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--K_max", type=int, default=8,
                    help="Max visit-dates for multi-visit modes.")
    ap.add_argument("--samples_root", default=SAMPLES_ROOT,
                    help="Override patient_samples root (e.g. patient_samples_K16)")
    ap.add_argument("--endpoint", default="mace",
                    help="Endpoint name (e.g. mace, alldeath, mi, stroke).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp_name", required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)

    args = ap.parse_args()

    # ===== mode -> dataset/model config =====
    mode = args.mode
    if mode in ("LP1", "M1"):
        K_input, shuffle_dt, train_random_K = 1, False, False
    elif mode == "LP2":
        K_input, shuffle_dt, train_random_K = args.K_max, False, False
    elif mode == "M2":
        K_input, shuffle_dt, train_random_K = args.K_max, False, True
    elif mode == "M3":
        K_input, shuffle_dt, train_random_K = args.K_max, True, True
    elif mode == "M2nt":
        # M2 but no time embedding (Set-Transformer style)
        K_input, shuffle_dt, train_random_K = args.K_max, False, True
    elif mode == "M2big":
        # Larger aggregator on multi-visit (test head capacity bottleneck)
        K_input, shuffle_dt, train_random_K = args.K_max, False, True

    is_linear = mode.startswith("LP")
    no_time = (mode == "M2nt")

    # ===== reproducibility =====
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ===== output =====
    out_dir = Path(RUNS_DIR) / f"{args.endpoint}{args.horizon}yr" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"=== {args.exp_name} ({mode}) seed={args.seed} ===")
    log(f"  K_input={K_input}  shuffle_dt={shuffle_dt}  "
        f"train_random_K={train_random_K}  is_linear={is_linear}")
    log(f"  args: {vars(args)}")

    # ===== data =====
    samples = Path(args.samples_root) / f"{args.endpoint}{args.horizon}yr"
    log(f"  samples_root: {samples}")
    ds_tr = PatientDataset(samples / "train.pkl", K_input, shuffle_dt,
                           train_random_K=train_random_K, seed=args.seed)
    ds_va = PatientDataset(samples / "val.pkl",   K_input, shuffle_dt,
                           train_random_K=False, seed=args.seed)
    ds_te = PatientDataset(samples / "test.pkl",  K_input, shuffle_dt,
                           train_random_K=False, seed=args.seed)
    log(f"  train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")

    pos = ds_tr.df["label"].sum()
    neg = len(ds_tr) - pos
    pos_weight = float(neg) / max(pos, 1)
    log(f"  train pos={pos} neg={neg} pos_weight={pos_weight:.2f}")

    dl_kw = dict(num_workers=args.num_workers, pin_memory=True,
                 collate_fn=collate)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       drop_last=True, **dl_kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, **dl_kw)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, **dl_kw)

    # ===== model =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device: {device}")
    if is_linear:
        model = LinearProbe(D_FEAT).to(device)
    else:
        model = MultiVisitTransformer(
            D_FEAT, d=args.agg_d,
            n_layers=args.agg_layers, n_heads=args.agg_heads,
            no_time=no_time).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  model={type(model).__name__}  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pw = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ===== train loop =====
    best_auc = -1.0
    best_epoch = -1
    bad = 0
    t0 = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in dl_tr:
            emb = batch["emb"].to(device, non_blocking=True)
            eye = batch["eye"].to(device, non_blocking=True)
            dt  = batch["dt"].to(device, non_blocking=True)
            mask= batch["mask"].to(device, non_blocking=True)
            y   = batch["y"].to(device, non_blocking=True)
            logit = model(emb, eye, dt, mask)
            loss = loss_fn(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * y.size(0); tr_n += y.size(0)
        sched.step()
        tr_loss /= max(tr_n, 1)
        va = evaluate(model, dl_va, device)
        log(f"  ep{epoch:02d}  tr_loss={tr_loss:.4f}  val_auc={va['auc']:.4f}  "
            f"val_ap={va['ap']:.4f}  val_brier={va['brier']:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}")
        history.append({"epoch": epoch, "tr_loss": tr_loss, **{f"val_{k}": v
                        for k, v in va.items() if k != "preds"}})
        if va["auc"] > best_auc:
            best_auc = va["auc"]; best_epoch = epoch; bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  early stop at epoch {epoch} (best ep{best_epoch} auc={best_auc:.4f})")
                break

    elapsed = time.time() - t0
    log(f"\n[done] elapsed={elapsed/60:.1f}min  best_val_auc={best_auc:.4f} (ep{best_epoch})")

    # ===== test eval (best ckpt) =====
    log(f"\n[test] loading best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    te = evaluate(model, dl_te, device)
    log(f"[test] auc={te['auc']:.4f}  ap={te['ap']:.4f}  "
        f"brier={te['brier']:.4f}  n={te['n']}  pos={te['n_pos']}")

    # Save preds + summary
    te["preds"].to_csv(out_dir / "test_preds.csv", index=False)
    summary = {
        "mode": mode, "exp_name": args.exp_name, "seed": args.seed,
        "horizon": args.horizon, "K_input": K_input,
        "best_val_auc": best_auc, "best_val_epoch": best_epoch,
        "test_auc": te["auc"], "test_ap": te["ap"], "test_brier": te["brier"],
        "test_n": te["n"], "test_n_pos": te["n_pos"],
        "n_params": n_params, "elapsed_min": elapsed / 60,
        "args": vars(args),
        "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[wrote] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

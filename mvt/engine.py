"""Training and prediction for MVT, with the settings used in the study."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from .data import loader, to_device

CONFIG = dict(
    microbatch=2, effective_batch=16,         # 2 patients per step, 8 accumulation steps
    max_epochs=8, patience=3, warmup_epochs=1,  # encoder frozen in epoch 1
    lr_new=1e-4, lr_encoder=3e-6, layer_decay=.9, weight_decay=.01,
    clip_grad=1., seed=1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def optimizer_for(model, cfg=CONFIG):
    """AdamW. New modules: lr 1e-4. Encoder: lr 3e-6 with layer-wise decay 0.9 from the last
    block downwards. Weight decay 0.01, none on biases and normalisation parameters."""
    top = len(model.encoder.blocks) + 1
    groups = {}
    for name, p in model.named_parameters():
        no_decay = p.ndim < 2 or name.endswith('bias')
        if name.startswith('encoder.'):
            short = name[len('encoder.'):]
            level = (int(short.split('.')[1]) + 1 if short.startswith('blocks.')
                     else top if short.startswith('norm.') else 0)
            lr = cfg['lr_encoder'] * cfg['layer_decay'] ** (top - level)
        else:
            lr = cfg['lr_new']
        groups.setdefault((lr, 0. if no_decay else cfg['weight_decay']), []).append(p)
    return torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()])


def metrics(frame):
    return dict(auroc=float(roc_auc_score(frame.y, frame.prob)),
                average_precision=float(average_precision_score(frame.y, frame.prob)),
                n=len(frame), events=int(frame.y.sum()))


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    rows = []
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            prob = torch.sigmoid(model(to_device(batch, device))).float().cpu().numpy()
        rows.append(pd.DataFrame(dict(pid=batch['pid'], y=batch['y'].numpy().astype(int), prob=prob)))
    return pd.concat(rows, ignore_index=True)


def fit(model, train_set, val_set, out_dir, device, cfg=CONFIG, workers=2):
    """Train with mixed precision and gradient accumulation; keep the checkpoint with the highest
    validation AUROC; stop after `patience` epochs without improvement."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    amp = device.type == 'cuda'
    validation = loader(val_set, cfg['microbatch'], workers=workers)
    opt = optimizer_for(model, cfg)
    steps = math.ceil(len(train_set) / cfg['effective_batch'])  # optimiser steps per epoch

    def rate(step):  # one-epoch linear warm-up, then cosine decay to zero at the last epoch
        warm = max(1, steps)
        if step < warm:
            return .05 + .95 * step / warm
        return .5 * (1 + math.cos(math.pi * min(1., (step - warm) / (cfg['max_epochs'] * steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., enabled=amp)
    y = np.array([r['y'] for r in train_set.records])
    pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum()], device=device)  # negatives / positives

    def save_best(epoch, m):
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        epoch=epoch, metrics=m, config=cfg), out / 'best.pt')

    model.set_epoch(1, cfg['warmup_epochs'])
    m = metrics(predict(model, validation, device))  # the initial model is the first reference
    best, bad, history = m['auroc'], 0, [dict(epoch=0, **m)]
    save_best(0, m)
    accum = cfg['effective_batch'] // cfg['microbatch']
    for epoch in range(1, cfg['max_epochs'] + 1):
        if bad >= cfg['patience']:
            break
        train_set.epoch = epoch
        batches = loader(train_set, cfg['microbatch'], shuffle=True, workers=workers)
        model.set_epoch(epoch, cfg['warmup_epochs'])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.
        for step, batch in enumerate(batches):
            b = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(b)
                loss = F.binary_cross_entropy_with_logits(logits, b['y'], pos_weight=pos_weight, reduction='sum')
            # average over the patients of the current effective batch
            group_start = (step // accum) * accum * cfg['microbatch']
            scaler.scale(loss / min(cfg['effective_batch'], len(train_set) - group_start)).backward()
            loss_sum += float(loss.detach())
            if (step + 1) % accum == 0 or step + 1 == len(batches):
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if torch.isfinite(norm):
                    sched.step()
        m = metrics(predict(model, validation, device))
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(train_set), **m))
        print(json.dumps(history[-1]), flush=True)
        if m['auroc'] > best:
            best, bad = m['auroc'], 0
            save_best(epoch, m)
        elif epoch > cfg['warmup_epochs']:
            bad += 1
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return history

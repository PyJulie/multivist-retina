"""Train MVT-1 or MVT-K.

    python train.py --train train.jsonl --val val.jsonl --weights retfound_dinov2.pth \
        --mode mvtk --max-visits 8 --out runs/mace3y_mvtk
"""
import argparse
import json
from pathlib import Path

import torch

from mvt import MVT
from mvt.data import PatientHistories
from mvt.encoder import retfound_dinov2
from mvt.engine import CONFIG, fit, seed_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True, help='training manifest (JSON Lines)')
    ap.add_argument('--val', required=True, help='validation manifest (JSON Lines)')
    ap.add_argument('--weights', required=True, help='RETFound-DINOv2 checkpoint')
    ap.add_argument('--image-root', default='.', help='folder that image paths are relative to')
    ap.add_argument('--mode', choices=('mvt1', 'mvtk'), default='mvtk')
    ap.add_argument('--max-visits', type=int, default=8, help='K: 8 for AlzEye, 4 for iKang')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(CONFIG['seed'])
    model = MVT(retfound_dinov2(args.weights)).to(device)
    latest = args.mode == 'mvt1'
    train_set = PatientHistories(args.train, args.image_root, args.max_visits, latest, training=True)
    val_set = PatientHistories(args.val, args.image_root, args.max_visits, latest)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'settings.json').write_text(json.dumps(dict(vars(args), **CONFIG), indent=2))
    fit(model, train_set, val_set, args.out, device, workers=args.workers)


if __name__ == '__main__':
    main()

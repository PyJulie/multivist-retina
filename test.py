"""Evaluate a trained MVT checkpoint on a test manifest.

    python test.py --test test.jsonl --checkpoint runs/mace3y_mvtk/best.pt \
        --mode mvtk --max-visits 8 --out runs/mace3y_mvtk/test_predictions.csv
"""
import argparse
import json

import torch

from mvt import MVT
from mvt.data import PatientHistories, loader
from mvt.encoder import retfound_dinov2
from mvt.engine import metrics, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', required=True, help='test manifest (JSON Lines)')
    ap.add_argument('--checkpoint', required=True, help='best.pt written by train.py')
    ap.add_argument('--image-root', default='.')
    ap.add_argument('--mode', choices=('mvt1', 'mvtk'), default='mvtk')
    ap.add_argument('--max-visits', type=int, default=8)
    ap.add_argument('--out', required=True, help='CSV of per-patient predictions')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MVT(retfound_dinov2(weights=None))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'], strict=True)
    model.to(device)
    test_set = PatientHistories(args.test, args.image_root, args.max_visits, latest_only=args.mode == 'mvt1')
    frame = predict(model, loader(test_set, batch=2, workers=args.workers), device)
    frame.to_csv(args.out, index=False)
    print(json.dumps(metrics(frame)))


if __name__ == '__main__':
    main()

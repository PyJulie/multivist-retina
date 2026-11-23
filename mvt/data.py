"""Patient imaging histories for MVT.

Input is a JSON Lines manifest with one patient per line, for example

    {"pid": "p001", "label": 1,
     "images": [{"path": "p001/2019-05-02_R.jpg", "eye": "R", "date": "2019-05-02"},
                {"path": "p001/2019-05-02_L.jpg", "eye": "L", "date": "2019-05-02"},
                {"path": "p001/2017-03-14_R.jpg", "eye": "R", "date": "2017-03-14"}]}

`eye` is R, L or U (unknown). The index date is the most recent visit date, and each image's
interval is the number of days between its acquisition date and the index date. Only the
`max_visits` most recent visit dates are kept (K = 8 in AlzEye, 4 in iKang).

MVT-K: during training the number of visit dates is sampled uniformly between one and the
number available (at most K), with a fixed seed per patient and epoch. At evaluation all kept
visit dates are used. MVT-1: all images from the most recent visit date only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize

IMAGE_SIZE = 224


def eye_code(value):
    value = str(value).strip().upper()
    return 0 if value in ('R', 'RIGHT', 'OD') else 1 if value in ('L', 'LEFT', 'OS') else 2


def read_image(path):
    """RGB image resized to 224 x 224, as uint8 (H, W, 3)."""
    with Image.open(path) as image:
        image.draft('RGB', (256, 256))
        image = image.convert('RGB')
        image.load()
        return np.array(Resize((IMAGE_SIZE, IMAGE_SIZE))(image), dtype=np.uint8)


class PatientHistories(Dataset):
    def __init__(self, manifest, image_root='.', max_visits=8, latest_only=False, training=False, seed=1):
        self.root, self.latest_only, self.training, self.seed = Path(image_root), latest_only, training, seed
        self.epoch = 0
        self.records = []
        with open(manifest, encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    self.records.append(self._record(json.loads(line), max_visits))

    @staticmethod
    def _record(row, max_visits):
        images = row['images']
        dates = pd.to_datetime([im['date'] for im in images])
        kept_dates = sorted(set(dates))[-max_visits:]
        keep = [i for i, d in enumerate(dates) if d in kept_dates]
        index_date = kept_dates[-1]
        return dict(pid=str(row['pid']), y=int(row['label']),
                    paths=[images[i]['path'] for i in keep],
                    eye=np.asarray([eye_code(images[i]['eye']) for i in keep], dtype=np.int64),
                    dt=np.asarray([(index_date - dates[i]).days for i in keep], dtype=np.float32),
                    date=np.asarray([kept_dates.index(dates[i]) for i in keep], dtype=np.int64),
                    n_dates=len(kept_dates))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        n = r['n_dates']
        if self.latest_only:
            keep = np.flatnonzero(r['date'] == n - 1)
        else:
            k = n
            if self.training:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
                k = int(rng.integers(1, n + 1))
            keep = np.flatnonzero(r['date'] >= n - k)
        return dict(x=np.stack([read_image(self.root / r['paths'][i]) for i in keep]),
                    pid=r['pid'], y=r['y'], eye=r['eye'][keep], dt=r['dt'][keep])


def collate(rows):
    """Concatenate the images of all patients; `groups` gives each patient's image range."""
    groups, offset = [], 0
    for r in rows:
        groups.append((offset, offset + len(r['eye'])))
        offset += len(r['eye'])
    return dict(x=torch.from_numpy(np.concatenate([r['x'] for r in rows])),
                eye=torch.from_numpy(np.concatenate([r['eye'] for r in rows])),
                dt=torch.from_numpy(np.concatenate([r['dt'] for r in rows])),
                groups=groups, pid=[r['pid'] for r in rows],
                y=torch.tensor([r['y'] for r in rows], dtype=torch.float32))


def loader(dataset, batch=2, shuffle=False, workers=2):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=False, collate_fn=collate,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(1000 + dataset.epoch))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

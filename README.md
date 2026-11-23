# Multi-visit transformer (MVT)

Code accompanying the manuscript *"Longitudinal oculomics improves cardiovascular risk and
mortality prediction"*.

The repository contains the model and its training and test code. The study data (AlzEye and
iKang) cannot be shared, so the code reads a generic manifest of image paths, laterality and
acquisition dates.

```
mvt/model.py     MVT: RETFound encoder, interaction modules C1-C3, patient-level readout
mvt/encoder.py   RETFound-DINOv2 ViT-Large encoder (224 x 224 input)
mvt/data.py      patient histories and visit selection for MVT-1 and MVT-K
mvt/engine.py    optimiser, learning-rate schedule, training loop and prediction
train.py         train MVT-1 or MVT-K
test.py          predict a test set and report AUROC and average precision
```

## Model

- **Encoder.** RETFound-DINOv2 ViT-Large (24 blocks) at 224 x 224 pixels. Positional embeddings
  of a checkpoint trained at another resolution are resampled by bicubic interpolation.
- **Interaction modules.** C1, C2 and C3 sit after encoder blocks 18, 21 and 24 and have separate
  parameters. Each keeps the classification token, average-pools the patch tokens to a 4 x 4
  grid (17 tokens per image), projects them to 768 dimensions and adds positional, laterality
  (right, left or unknown) and time embeddings. The time embedding encodes the interval to the
  index date with sine and cosine at 1, 7, 30, 90, 180, 365, 730 and 1,825 days, followed by a
  two-layer projection.
- **Attention inside each module.** Within-image spatial attention, then same-eye cross-visit
  attention, then same-visit bilateral attention. Cross-image attention uses only images with
  known laterality and adds a learned per-head bias of the signed interval between the two
  images. The update returns to the encoder tokens through a linear projection, bilinear
  upsampling to the patch grid and a learnable residual scale initialised at 0.01.
- **Patient-level readout.** Eight learned queries, two cross-attention layers, mean over the
  queries, layer normalisation, dropout and a linear layer giving one logit per patient.
- All attention layers use pre-normalisation, 12 heads, a feed-forward expansion of 4 and
  dropout of 0.1.

## Training settings

| Setting | Value |
|---|---|
| Visits | MVT-1: all images from the most recent visit date. MVT-K: up to the K most recent visit dates (K = 8 in AlzEye, 4 in iKang). During training the number of visit dates is sampled uniformly between one and the number available. |
| Encoder | frozen in epoch 1, fine-tuned from epoch 2 |
| Optimiser | AdamW; learning rate 1e-4 for new modules and 3e-6 for the encoder, with layer-wise decay 0.9; weight decay 0.01 |
| Schedule | one-epoch linear warm-up, cosine decay, at most 8 epochs |
| Early stopping | after 3 epochs without improvement in validation AUROC; the checkpoint with the highest validation AUROC is kept |
| Batch | 16 patients (2 per step, 8 accumulation steps), mixed precision, gradient clipping at 1.0 |
| Loss | binary cross-entropy, positive class weighted by the negative-to-positive ratio of the training set |
| Seed | 1 |

The values are set in `CONFIG` in `mvt/engine.py`.

## Input format

One JSON Lines manifest per split, one patient per line:

```json
{"pid": "p001", "label": 1,
 "images": [{"path": "p001/2019-05-02_R.jpg", "eye": "R", "date": "2019-05-02"},
            {"path": "p001/2019-05-02_L.jpg", "eye": "L", "date": "2019-05-02"},
            {"path": "p001/2017-03-14_R.jpg", "eye": "R", "date": "2017-03-14"}]}
```

`eye` is `R`, `L` or `U` (unknown). The index date is the most recent visit date, and image paths
are relative to `--image-root`.

## Usage

```bash
pip install -r requirements.txt

python train.py --train train.jsonl --val val.jsonl --weights retfound_dinov2.pth \
    --mode mvtk --max-visits 8 --out runs/mvtk

python test.py --test test.jsonl --checkpoint runs/mvtk/best.pt \
    --mode mvtk --max-visits 8 --out runs/mvtk/test_predictions.csv
```

Use `--mode mvt1` for the single-visit model. The RETFound-DINOv2 weights are released by the
RETFound authors and are not included here. Training needs a GPU; the code was run on a single
48 GB GPU.

## Licence

See `LICENSE`.

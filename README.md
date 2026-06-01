# Multi-visit transformer — training and evaluation code

Reference code accompanying the manuscript
*"Longitudinal retinal imaging predicts cardiovascular risk that no single image fully captures."*

This repository contains only the **model / training code for the compared methods** and
the **evaluation (test) code**. Data-construction, cohort-building, feature-extraction,
ablation and analysis scripts are intentionally **not** included, so the repository carries
no dataset field definitions, no image paths, and no machine-specific configuration. The
models operate on pre-extracted, cached image features (one frozen 1024-d CLS vector per
image); producing those features and the per-patient sample tables is out of scope here.

```
training/
  train_multivisit.py             # single-visit (MVT-1) vs multi-visit (MVT-K) transformer
  train_multivisit_multitask.py   # MVT-Joint: one shared visit aggregator, multiple CV-endpoint heads
  train_multivisit_survival.py    # MVT-Surv: discrete-time survival variant
evaluation/
  significance.py                 # seed-ensembled DeLong test + paired bootstrap 95% CI (M2 vs M1)
  collect_results.py              # aggregate per-run test summaries into AUROC / gain tables
```

## Model

Each image is one input token: a frozen RETFound ViT-Large CLS feature (1024-d) is linearly
projected to 256-d, with a learned laterality (L/R) embedding and a sinusoidal
image-to-index time-delta embedding added. A learned CLS token is prepended and the
sequence is processed by a 2-layer Transformer encoder (4 heads, 4x feed-forward, dropout
0.1, norm-first); the CLS output gives one prediction per patient. Training: AdamW
(lr 1e-4, weight decay 1e-4), cosine schedule, batch size 128, gradient clip 1.0,
BCE-with-logits with positive-class weighting, number of visit-dates sampled uniformly in
[1, 8] per patient, early stopping on validation AUROC. The multi-task variant shares the
visit aggregator across endpoints; the survival variant replaces the head with a 20-bin
(three-month) discrete-time sigmoid-hazard head.

## Input contract

The training scripts read per-split sample tables (one row per patient) with columns:
`patient_id`, `label`, `input_paths` (image identifiers for that patient's visits),
`input_dt_days` (image-to-index time deltas), `input_lat` (laterality), and a feature
store keyed by image identifier. Building these tables from a specific cohort is
deliberately left to the user. Evaluation reads per-run `test_preds.csv` (`pid, y, prob`)
and `summary.json`.

## Modes (in `train_multivisit.py`)

`M1` single-visit (K=1) · `M2` multi-visit · plus control modes used in the paper's
ablations. Run scripts for the ablations / K-sweep are not included.

## Environment

See `requirements.txt`. Paths are passed as environment variables / CLI arguments
(`${RUNS_DIR}`, `${SAMPLES_DIR}`, `${CACHE_DIR}`, ...); no paths are hard-coded.

## Data availability

The study datasets are not publicly redistributable and are governed by their data
controllers. Foundation-model weights are available from their original providers.

## License

Released for **non-commercial research use** under CC BY-NC 4.0, matching the licensing of
the RETFound foundation models this work builds on. See `LICENSE`.

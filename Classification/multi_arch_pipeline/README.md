# Multi-Architecture Unlearning Pipeline

Companion experiments to extend the main MUKSB results in
`../run_pipeline.sh` (ResNet-18 / CIFAR-10) across **multiple network
architectures** for the paper supplementary.

For each architecture we compare two methods on the same 4 500-sample
random forget set:

| Method     | Description                                                   |
|------------|---------------------------------------------------------------|
| `MUKSB`    | Our unlearning method (proposed)                              |
| `retrain`  | Strict gold-standard baseline — retrain from scratch on the   |
|            | retain set only (no forget samples)                           |

## Layout

```
multi_arch_pipeline/
├── README.md
├── pretrain.sh              # one-shot pretrain on full CIFAR-10
├── run_unlearn.sh           # generic runner (MUKSB + retrain) — sourced by wrappers
├── run_resnet18_cifar10.sh  # ResNet-18 wrapper (uses existing checkpoint)
├── run_vgg16_cifar10.sh     # VGG-16 BN wrapper (pretrains if needed)
├── run_swin_t_cifar10.sh    # Swin-T wrapper (fine-tunes ImageNet weights)
├── run_all.sh               # runs every arch + aggregates results
└── aggregate_results.py     # collects per-seed eval into one CSV
```

Results land under `../results_multi_arch/<dataset>/<arch>/<method>/<forget_tag>/seed<N>/`.

## Quick start

Run everything (ResNet-18, VGG-16, and Swin-T, MUKSB + retrain):

```bash
cd multi_arch_pipeline
bash run_all.sh
```

Run a single architecture:

```bash
bash run_resnet18_cifar10.sh
bash run_vgg16_cifar10.sh       # auto-pretrains VGG-16 first
bash run_swin_t_cifar10.sh      # auto-fine-tunes Swin-T (ImageNet → CIFAR-10)
```

### Swin-T notes

Swin-T expects 224×224 inputs, so [models/SwinT.py](../models/SwinT.py)
wraps the torchvision model and upsamples 32×32 CIFAR inputs on the
fly. Defaults: ImageNet-pretrained init, batch size 128, LR 1e-3,
3 seeds × 3 GPUs (Swin-T runs slower than the convnets).  Set
`MUKSB_SWIN_PRETRAINED=0` to start from random init instead, and
override `SEEDS=(1 2 3 4 5) GPUS=(0 1 2 3 4) EPOCHS=(30 30 30 30 30)`
to match the convnet seed count.

Aggregate any time (does not need all runs to be finished):

```bash
python aggregate_results.py \
    --results_root ../results_multi_arch \
    --out_csv      multi_arch_summary.csv
```

## Hyperparameters

The wrappers mirror `../run_pipeline.sh` so the new numbers are
directly comparable with the main table:

| Setting              | Value                  |
|----------------------|------------------------|
| Forget set           | 4 500 random samples   |
| Unlearn epochs       | 30                     |
| Unlearn LR           | 0.03 (convnet) / 1e-3 (Swin-T) |
| γ (KS retain weight) | 0.5                    |
| α (noise)            | 0.2                    |
| Batch size           | 512 (convnet) / 128 (Swin-T)    |
| Seeds                | 1, 2, 3, 4, 5 (convnet) / 1, 2, 3 (Swin-T) |
| GPUs                 | 0, 1, 2, 3, 4 (convnet) / 0, 1, 2 (Swin-T) |

Override at the call-site by editing the wrapper or exporting env vars
before `source`-ing `run_unlearn.sh`.

## Adding a new architecture

1. Make sure the arch is in `model_dict` (`../models/__init__.py`).
2. Copy `run_vgg16_cifar10.sh` → `run_<arch>_<dataset>.sh`, update
   `ARCH`, `CKPT_DIR`, and (optionally) pre-training hyperparams.
3. Add it to the `case` in `run_all.sh`.

The aggregator picks up new (arch × method) combinations automatically.

## Reading the CSV

`multi_arch_summary.csv` contains one row per seed plus a final
mean ± std row per (arch, method). The columns to quote in the
supplementary table:

- `acc_retain`, `acc_forget`, `acc_val`, `acc_test`
- `mia` — paper MIA (logit-distinguishability)
- `svc_mia_forget_efficacy_confidence`, `svc_mia_forget_efficacy_entropy`

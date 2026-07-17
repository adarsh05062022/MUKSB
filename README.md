# MUKSB: Magnitude-Aware Kalai–Smorodinsky Bargaining for Machine Unlearning

MUKSB frames machine unlearning as a **two-player bargaining game** between the
*retain* objective (don't forget what the model should keep) and the *forget*
objective (erase the target class / concept / capability). Instead of summing
or averaging the retain and forget gradients — which lets whichever gradient
has larger magnitude dominate the update — MUKSB computes the
**Kalai–Smorodinsky (KS) bargaining solution** between the two (unit-normalized,
magnitude-rescaled) gradients at every step. This repository is the reference
implementation, benchmarked against a Nash-bargaining baseline (**MUNBa**) and
a broad set of prior unlearning methods, across five model families:

| Domain | Directory | Backbone(s) |
|---|---|---|
| Image classification | [`Classification/`](Classification) | ResNet-18/34, VGG, Swin-T on CIFAR-10/100, ImageNet, CelebA |
| Class-conditional diffusion | [`DDPM/`](DDPM) | DDPM/DDIM on CIFAR-10, STL-10 |
| Text-to-image diffusion | [`SD/`](SD) | Stable Diffusion v1.4 (CompVis/LDM) |
| Text-to-image diffusion | [`SDXL/`](SDXL) | Stable Diffusion XL (Diffusers) |
| Text-to-image diffusion | [`SD3/`](SD3) | Stable Diffusion 3 Medium (Diffusers, flow-matching DiT) |
| Instruction-guided image editing | [`IP2P/`](IP2P) | InstructPix2Pix |

Unlearning targets covered by the provided scripts include single-class /
multi-class forgetting (classification, DDPM), object-concept forgetting,
celebrity/identity forgetting, and NSFW-concept removal (SD / SDXL / SD3 /
IP2P).

## How MUKSB works (short version)

For a training step with retain-loss gradient `g_r` and forget-loss gradient
`g_f` (the forget loss is usually negated/ascended so both gradients "pull" in
directions the bargaining solution has to reconcile):

1. Normalize both gradients to unit vectors and compute the angle between them.
2. Solve for the KS bargaining point on the segment between the two unit
   gradients — the point that gives each objective a "fair" share of the
   combined disagreement, rather than letting the larger-magnitude gradient
   win (as a naive sum does).
3. Rescale the resulting direction by an effective magnitude derived from
   `‖g_r‖` and `‖g_f‖` (the harmonic-mean-style term in `ks_step`), so the
   update size stays sensible.
4. Apply the result as the parameter update over all trainable parameters.

The core `ks_step` routine is duplicated (kept in sync deliberately, not
imported) across `Classification/unlearn/MUKSB.py`, `SD/MUKSB_*.py`,
`SDXL/MUKSB_nsfw.py`, `SD3/MUKSB_nsfw.py`, and `IP2P/MUKSB_nsfw_i2i.py` so
that each domain's training script stays standalone/runnable on its own.

## Repository layout

```
Classification/     ResNet/VGG/Swin-T unlearning on CIFAR/ImageNet/CelebA
├── unlearn/         All unlearning methods (see table below), incl. MUKSB.py, MUNBa.py
├── models/          Architectures
├── trainer/         Train/val loops
├── pruner/          OMP / SynFlow pruning utilities used by prune-based baselines
├── evaluation/       MIA / SVC-MIA / forgetting-quality metrics
├── main_train.py     Train a base model
└── main_forget.py    Run an unlearning method against a base checkpoint

DDPM/                Class-conditional DDPM unlearning (CIFAR-10 / STL-10)
├── train.py, sample.py, fim.py, classifier_evaluation.py

SD/                  Stable Diffusion v1.4 (CompVis/LDM) unlearning
├── MUKSB_nsfw.py, MUKSB_celebrity.py, MUKSB_cls.py   Entry points per target
├── DiMRA_nsfw.py, DiMRA_cls.py                        Relearning-attack evaluation
├── train_scripts/, ldm/, configs/, src/                Model + training internals
└── eval_scripts/, Evaluation/                          FID / classifier / CLIP-NSFW eval

SDXL/, SD3/          Diffusers-based counterparts of SD/ (NSFW-removal focus)
├── MUKSB_nsfw.py, gen_images.py, train_scripts/, Evaluation/

IP2P/                InstructPix2Pix NSFW-removal (image-to-image unlearning)
├── MUKSB_nsfw_i2i.py, generate_nsfw_i2i.py, eval_nsfw_i2i.py

compute_asr_q16.py       Attack Success Rate via the Q16 classifier (CASteer/SAFREE/RECE protocol)
compute_clip_attacks.py  CLIP image–text similarity for attack-benchmark images
```

## Baselines implemented

Alongside `MUKSB`, the `Classification/unlearn/` and the SD/SDXL/SD3/IP2P
`MUNBa.py`-style modules provide these comparison methods: `GA`, `GA_l1`,
`GA_prune`, `GA_prune_bi`, `RL`, `RL_proximal`, `FT`, `FT_l1`, `FT_prune`,
`FT_prune_bi`, `fisher`, `fisher_new`, `Wfisher`, `boundary_expanding`,
`boundary_shrink`, `SHs`, `SalUn` ([Fan et al., ICLR 2024](https://arxiv.org/abs/2310.12508)),
`IU` ([Izzo et al., AISTATS 2021](https://arxiv.org/abs/2012.14913)), and
`MUNBa` (Nash-bargaining baseline). `retrain` (exact/gold-standard retraining)
is included where feasible.

## Setup

```bash
conda env create -f environment.yml
conda activate munba3
```

Notes:

- `DDPM/` has its own `requirements.txt` (it is adapted from a separate,
  older Saliency-Unlearning/DDIM codebase — see [`DDPM/README.md`](DDPM/README.md)).
- `SD3/` scripts were developed against newer `diffusers`/`transformers`
  versions than the root `environment.yml` pins for `SD`/`SDXL`. If you hit
  dependency conflicts, create a second environment for `SD3` with an
  up-to-date `diffusers` (see the version-specific notes at the top of
  `SD3/MUKSB_nsfw.py`).
- `SD/` uses the original CompVis/`ldm` codebase and expects a CompVis-format
  checkpoint (e.g. `sd-v1-4-full-ema.ckpt`) plus `SD/configs/stable-diffusion/v1-inference.yaml`.
  `SDXL`/`SD3`/`IP2P` use HuggingFace `diffusers` pipelines/model IDs instead
  and don't need a local checkpoint download.
- Datasets and pretrained/unlearned checkpoints are **not** included in this
  repository (see `.gitignore`) — point the scripts below at your own copies.

## Quickstart

All example paths below are placeholders (`/path/to/...`) — replace them
with your own dataset/checkpoint locations; several scripts currently ship
with the original authors' cluster paths as defaults, so double-check
`--help` output before relying on unset flags.

### Classification

```bash
# 1. Unlearn with MUKSB, starting from a pretrained base checkpoint
python Classification/main_forget.py --unlearn MUKSB --class_to_replace 1 \
    --dataset cifar10 --arch resnet18 --gpu 0 \
    --mask /path/to/pretrained.pth --save_dir results/muksb_cls1

# 2. Compare against the Nash-bargaining baseline
python Classification/main_forget.py --unlearn MUNBa --class_to_replace 1 \
    --dataset cifar10 --arch resnet18 --gpu 0 \
    --mask /path/to/pretrained.pth --save_dir results/munba_cls1
```

Note: despite the flag name, `--mask` here is the path to the pretrained
checkpoint to unlearn from (inherited naming from the codebase this was
built on).

### Stable Diffusion v1.4 — NSFW concept removal

```bash
python SD/MUKSB_nsfw.py \
    --forget_path /path/to/nude \
    --remain_path /path/to/with_dress \
    --train_method full --epochs 5 --lr 1e-5 --device 0
```

`SD/MUKSB_celebrity.py` and `SD/MUKSB_cls.py` follow the same pattern for
celebrity-identity and object-class forgetting respectively.

### SDXL / SD3 — NSFW concept removal

```bash
python SDXL/MUKSB_nsfw.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --forget_path /path/to/nude --remain_path /path/to/with_dress \
    --train_method full --epochs 5 --lr 1e-5 --device 0

python SD3/MUKSB_nsfw.py \
    --model_id stabilityai/stable-diffusion-3-medium-diffusers \
    --forget_path /path/to/nude --remain_path /path/to/with_dress \
    --train_method full --epochs 5 --lr 1e-5 --device 0
```

### InstructPix2Pix — NSFW concept removal (image-to-image)

```bash
python IP2P/MUKSB_nsfw_i2i.py --epochs 5 --device 0
```

### DDPM — class-conditional forgetting

See [`DDPM/README.md`](DDPM/README.md) for the full train → generate-saliency →
unlearn pipeline (adapted from the original Saliency-Unlearning DDPM codebase).

## Evaluation

- **Membership inference (MIA)** — `Classification/evaluation/MIA.py`, `SVC_MIA.py`.
- **Forget/retain accuracy, FID, classifier accuracy** — `SD/eval_scripts/`
  (`compute_fid.py`, `compute_ua_imagenette_csv.py`, `imageclassify.py`).
- **NSFW-removal quality** — `*/Evaluation/nsfw/` (CLIP-NSFW score) plus
  [NudeNet](https://github.com/notAI-tech/NudeNet).
- **Adversarial robustness / red-teaming**:
  - `compute_asr_q16.py` — Attack Success Rate via the Q16 CLIP classifier,
    following the CASteer / SAFREE / RECE evaluation protocol.
  - `compute_clip_attacks.py` — CLIP image–text similarity on attack-generated images.
  - `SD/DiMRA_nsfw.py`, `SD/DiMRA_cls.py` — **relearning attack**: fine-tunes
    the *unlearned* model on a benign auxiliary set to test whether the
    forgotten capability re-emerges (["Towards Irreversible Machine
    Unlearning for Diffusion Models"](https://arxiv.org/abs/2512.03564)).

## Citation

If you use this code, please cite:

```bibtex
@article{muksb2026,
  title   = {<paper title>},
  author  = {<authors>},
  journal = {<venue, year>},
}
```

## License

<!-- TODO: add a top-level LICENSE (e.g. MIT) before release. -->
This repository does not currently declare a license for its own code.
Vendored subcomponents keep their original licenses: `DDPM/` (see
[`DDPM/LICENSE`](DDPM/LICENSE)) and `SD/src/taming-transformers/` (see
[`License.txt`](SD/src/taming-transformers/License.txt)).

## Acknowledgements

Built on top of / evaluated against: [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion),
[taming-transformers](https://github.com/CompVis/taming-transformers),
[DDIM](https://github.com/ermongroup/ddim),
[Selective Amnesia](https://github.com/clear-nus/selective-amnesia),
[SalUn](https://arxiv.org/abs/2310.12508), and the **MUNBa** Nash-bargaining
unlearning baseline.

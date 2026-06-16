# MUKSB — Object Concept Removal (Dog / Car / Bicycle)

Handoff doc for the object-removal experiment. Goal: show MUKSB generalizes
beyond NSFW + Imagenette class unlearning to **generic object concepts**
(dog, car, bicycle) while preserving generation quality.

> Status as of last session: **scaffolding partly built, nothing run yet.**
> See "Progress" below for exactly what exists and what's left.

---

## 0. Environment (READ FIRST)

| Thing | Value |
|-------|-------|
| Python | `/storage/s25017/miniconda3/envs/munba3/bin/python` (conda env **munba3**, py3.10, torch 2.10+cu128) |
| Working dir | `/scratch/s25017/MUKSB/SD` |
| `python` on PATH | **NOT available** — always call the munba3 interpreter by full path (or `conda activate munba3`) |
| GPUs | 8× H200 (143 GB each) but **heavily loaded — ~70–85% memory used by other jobs**. Pick the freest with `nvidia-smi`. |
| SD v1.4 ckpt | `models/ldm/sd-v1-4-full-ema.ckpt` |
| CompVis config | `configs/stable-diffusion/v1-inference.yaml` |
| Datasets root | `/storage/s25017/Datasets` (writable) |
| Generated train images go to | `/storage/s25017/Datasets/object_removal/<concept>/{forget,retain}/` |
| COCO prompts (for FID/CLIP) | `prompts/coco_5k.csv`, `prompts/coco_30k.csv` (cols: `case_number,prompt,...`) |

Quick env sanity check:
```bash
PY=/storage/s25017/miniconda3/envs/munba3/bin/python
$PY -c "import torch,diffusers,transformers,clip,torchmetrics; print('ok', torch.cuda.device_count(),'gpus')"
```

---

## 1. Design decisions (already locked with the user)

1. **Training data = generated from SD v1.4** (SalUn-style, self-contained).
   No real dog/car/bicycle dataset on disk. We generate forget images from the
   forget prompts and retain images from the retain prompts using vanilla SD v1.4,
   then MUKSB unlearns on those.
2. **UA classifier = ResNet50 (ImageNet weights)**, but each concept maps to a
   **GROUP** of ImageNet indices (not one index — "dog/car/bicycle" each span
   many classes). See the index groups in §4.
3. **MUKSB only for now.** ESD / SalUn / MUNBa baseline columns are a later TODO.
4. **Build + smoke-test, then launch.** Don't kick off the full multi-hour
   generation/training/eval on the shared cluster until a tiny end-to-end smoke
   test (1 concept, ~8 imgs, 1 epoch) passes and the user confirms.

---

## 2. How MUKSB object removal works (the method — UNCHANGED core)

Identical Magnitude-Aware Kalai–Smorodinsky bargaining update as
`MUKSB_nsfw.py` / `MUKSB_cls.py`. Only data + the anchor prompt differ:

- **Forget loss**: `MSE( eps(forget_img | concept_prompt),
  eps(forget_img | anchor_prompt).detach() ) * beta`
  → steers the concept-conditioned prediction toward a neutral "random label"
  (`anchor_prompt`, default `"a photo"`).
- **Retain loss**: standard LDM denoising loss on retain images.
- **Update**: KS bargaining merge of `(grad_retain, grad_forget)` — `ks_step()`.

Each generated image carries the **prompt that generated it** as its caption
(via `manifest.csv`), so forget/retain conditioning matches image content.

---

## 3. Progress — what exists vs what's left

### ✅ DONE

- **`prompts/objects/make_object_prompts.py`** — reproducible prompt builder. Already run; produced:
  - `prompts/objects/{dog,car,bicycle}_forget.csv` — 200 rows each
  - `prompts/objects/{dog,car,bicycle}_retain.csv` — 200 rows each (generic non-target: cat/horse/bird/church/mountain/house/tree/person/boat/flower)
  - `prompts/objects/{dog,car,bicycle}_eval.csv`   — 100 rows each (held-out phrasings for UA)
  - CSV schema: `case_number,prompt,evaluation_seed,concept`
  - Re-run / rescale: `$PY prompts/objects/make_object_prompts.py --n_forget 300 --n_eval 100`
- **`train_scripts/dataset.py`** — added `ObjectDataset` + `setup_object_data(...)`
  (manifest-aware: reads `<dir>/manifest.csv` mapping `filename→prompt`; returns
  `{"jpg": HWC float in [-1,1], "txt": prompt}`, same contract as the NSFW datasets).
- **`MUKSB_object.py`** — training entry, adapted from `MUKSB_nsfw.py`
  (KS core copied verbatim). New args: `--concept`, `--anchor_prompt`,
  `--forget_path`, `--remain_path`, `--forget_caption`, `--remain_caption`.
  Output run tag: `compvis-obj_<concept>-MUKSB-method_full-lr_<lr>_E<ep>_U<n>_obj`
  (→ `diffusers-obj_...-epoch_<k>.pt` after conversion; the `compvis→diffusers`
  rename in `savemodelDiffusers` requires the tag to start with `compvis`).
  Prints `RUN_TAG=...` on stdout for the orchestrator to parse.

### ⬜ TODO (next session, in order)

1. **`Evaluation/objects/generate_objects_multigpu.py`** — prompt-CSV-driven
   multi-GPU generator (adapt `Evaluation/imagenette/generate_imagenette_multigpu.py`).
   - Reuse its `load_pipeline()` (loads diffusers SD v1.4 + optional UNet `.pt`;
     empty `model_path` → vanilla SD v1.4) and `get_gpu_slice()`.
   - Driven by a prompts CSV. Build a flat job list
     `(case_number, k, prompt, seed+k)` for `k in range(n_per_prompt)`; partition
     across GPUs; save as `{case_number:05d}_{k}.png` (filename MUST start with
     `case_number` — UA/CLIP parse `fname.split("_")[0]`).
   - After all workers finish, the launcher writes **`manifest.csv`**
     (`filename,prompt`) by mapping each produced file's case_number→prompt from
     the CSV (race-free; no per-worker manifest needed).
   - Expose `launch_multigpu(prompts_csv, output_dir, n_per_prompt, gpu_ids,
     model_path="", guidance_scale=7.5, image_size=512, ddim_steps=50, batch_size=4)`.
   - Used for: (a) training forget/retain images, (b) before/after eval images
     for the forgotten concept, (c) retain-quality images from COCO prompts.
2. **`Evaluation/objects/compute_ua_objects.py`** — ResNet50 group-UA
   (adapt `compute_ua_imagenette.py`). UA = % of forget-eval images whose top-1
   (and top-5) prediction is **NOT** in the concept's ImageNet index group (§4).
3. **`Evaluation/objects/compute_fid_objects.py`** — FID (feature=64, like
   `compute_fid_imagenette.py`) between the unlearned model's retain/COCO
   generations and a reference set. Make `--ref_dir` an arg; default reference =
   the **vanilla SD v1.4 generations on the same prompts** (measures quality
   preservation = "FID increase"). Optionally also support real COCO images.
4. **`Evaluation/objects/compute_clip_objects.py`** — CLIP image↔prompt cosine
   (copy `Evaluation/nsfw/compute_clip_nsfw.py` almost verbatim; it already reads
   `case_number→prompt` from a CSV and uses `clip` ViT-B/32).
5. **`Evaluation/objects/eval_objects.py`** — per-model orchestrator: given a
   model `.pt` + concept, generate eval images → UA + FID + CLIP → write
   `eval_summary.json`. Mirror `Evaluation/imagenette/eval_imagenette.py`.
6. **`run_object_removal.sh`** — top-level pipeline with a `SMOKE=1` mode.
   Stages: (0) generate train forget/retain imgs → (1) train MUKSB per concept →
   (2) generate before (SD v1.4) + after (unlearned) eval imgs + retain-quality
   imgs → (3) UA/FID/CLIP → (4) aggregate table. `PY` = munba3 interpreter.
7. **Smoke test**: dog, ~8 forget + 8 retain imgs, `--epochs 1 --batch_size 4`,
   ~8 eval imgs, run UA on them. Confirm end-to-end, then hand off launch commands.

---

## 4. UA ImageNet index groups (for `compute_ua_objects.py`)

ResNet50 `ResNet50_Weights.DEFAULT` (1000-class). A generated image "still is the
concept" iff its top-1 (UA) / any top-5 (UA5) prediction is in the group below.

```python
CONCEPT_IMAGENET_GROUPS = {
    # all ImageNet dog synsets (Stanford Dogs range) — 151..268 inclusive
    "dog": list(range(151, 269)),
    # passenger-car-ish classes
    "car": [436, 468, 511, 609, 627, 656, 661, 705, 717, 734, 751, 817],
    #       beach wagon, cab, convertible, jeep, limo, minivan, Model T,
    #       passenger car, pickup, police van, racer, sports car
    "bicycle": [444, 671],   # tandem/bicycle-built-for-two, mountain bike
}
```
> These are starting groups — verify a couple against
> `ResNet50_Weights.DEFAULT.meta["categories"]` and tweak (esp. `car`) if a
> baseline SD v1.4 "car" image lands outside the group.

---

## 5. Evaluation targets (from the spec)

| Metric | What | How |
|--------|------|-----|
| **UA ↑** | unlearning accuracy | % of forget-concept eval images NOT classified as the concept (ResNet50 group, §4) |
| **FID ↓** | retain quality | FID(feature=64) of retain/COCO gens vs reference (default: vanilla SD v1.4 gens) |
| **CLIP ↑** | prompt following | CLIP image↔prompt cosine on retain/COCO gens |

Final table (per concept + average), MUKSB now; ESD/SalUn/MUNBa later:

```
Forget Concept | FID ↓ | CLIP ↑ | UA ↑
Dog / Car / Bicycle / Average
```

Qualitative figure: same prompt (e.g. "a golden retriever sitting on grass")
side-by-side, vanilla SD v1.4 vs MUKSB-unlearned.

---

## 6. Reference files studied (source of truth for adaptation)

- Training: `MUKSB_nsfw.py` (folder+caption, pseudo prompt), `MUKSB_cls.py` (Imagenette classes)
- Data: `train_scripts/dataset.py` (`setup_nsfw_data`, now also `setup_object_data`)
- Model save/convert: `train_scripts/convertModels.py` (`savemodelDiffusers`)
- Generation: `Evaluation/imagenette/generate_imagenette_multigpu.py`
- Eval: `Evaluation/imagenette/eval_imagenette.py`, `compute_ua_imagenette.py`,
  `compute_fid_imagenette.py`; `Evaluation/nsfw/compute_clip_nsfw.py`
- This repo has a **code-review-graph MCP knowledge graph** — use its tools
  (`query_graph`, `semantic_search_nodes`, etc.) before Grep/Read (see `CLAUDE.md`).

---

## 7. Example commands (once §3 TODO is built)

```bash
PY=/storage/s25017/miniconda3/envs/munba3/bin/python
cd /scratch/s25017/MUKSB/SD

# (0) generate forget + retain training images for DOG (vanilla SD v1.4)
$PY Evaluation/objects/generate_objects_multigpu.py \
    --prompts_csv prompts/objects/dog_forget.csv \
    --output_dir  /storage/s25017/Datasets/object_removal/dog/forget \
    --n_per_prompt 1 --gpu_ids 0
$PY Evaluation/objects/generate_objects_multigpu.py \
    --prompts_csv prompts/objects/dog_retain.csv \
    --output_dir  /storage/s25017/Datasets/object_removal/dog/retain \
    --n_per_prompt 1 --gpu_ids 0

# (1) train MUKSB to forget DOG
$PY MUKSB_object.py --concept dog \
    --forget_path /storage/s25017/Datasets/object_removal/dog/forget \
    --remain_path /storage/s25017/Datasets/object_removal/dog/retain \
    --anchor_prompt "a photo" --epochs 5 --lr 1e-5 --batch_size 4 --device 0

# (2+3) evaluate (generate before/after + UA/FID/CLIP) — via eval_objects.py
# (full pipeline) — via run_object_removal.sh  (SMOKE=1 for the smoke test)
```

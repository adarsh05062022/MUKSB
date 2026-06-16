# CIFAR-10 / ResNet-18 — 10% Random Forgetting

Self-contained benchmark folder. Runs **9 unlearning methods × N seeds** on
CIFAR-10 / ResNet-18 with **10% random forgetting** (5,000 of 50,000 training
samples), then aggregates the per-seed results into a Table-1 style summary.

All scripts, checkpoints, logs, and results live **inside this folder**.

## Methods

| Folder name | `--unlearn` key      | Paper method            |
|-------------|----------------------|-------------------------|
| `retrain`   | `retrain`            | Retrain (oracle)        |
| `FT`        | `FT`                 | Fine-tune               |
| `GA`        | `GA`                 | Gradient Ascent         |
| `IU`        | `IU`                 | Influence Unlearning    |
| `BE`        | `boundary_expanding` | Boundary Expanding      |
| `l1sparse`  | `FT_l1`              | ℓ1-sparse fine-tuning   |
| `SalUn`     | `SalUn`              | Saliency Unlearning     |
| `MUNBa`     | `MUNBa`              | Nash bargaining         |
| `MUKSB`     | `MUKSB`              | **Ours (KS bargaining)**|

**Shared saliency mask:** `SalUn`, `MUNBa` and `MUKSB` all update the **same**
subset of weights — one gradient-saliency mask at density `MASK_DENSITY`
(default 0.5, i.e. top-50% of params by `|∇L_forget|`). The driver generates one
mask per seed (`masks/seed<N>/with_0.5.pt`, via `generate_mask.py`) and feeds it
to all three with `--path`, so their comparison is like-for-like. The other
methods (FT/GA/IU/BE/ℓ1/retrain) are unmasked. Change the density with
`MASK_DENSITY=0.3 bash run_all_methods.sh ...`.

## Layout

```
cifar10_random_forget/
├── run_all_methods.sh        # driver: 9 methods × N seeds
├── aggregate_results.py      # mean±std + Avg.Gap table + per-run JSON
├── README.md
├── masks/seed<N>/with_0.5.pt       # shared saliency mask (SalUn/MUNBa/MUKSB)
├── checkpoints/<method>/seed<N>/   # model + <key>eval_result.pth.tar + epoch_metrics.json
├── logs/<method>_seed<N>.log       # full stdout/stderr per run (+ mask_seed<N>.log)
└── results/
    ├── <method>/seed<N>/epoch_metrics.json   # per-epoch status (mirrored each run)
    ├── per_run/<method>_seed<N>.json         # final metrics + every epoch (combined)
    ├── all_runs.json                         # all runs in one file
    ├── per_seed_metrics.csv
    └── summary_mean_std.csv
```

### Per-epoch status JSON (like MUKSB)

Every method writes a live `epoch_metrics.json` to its run dir during training
— one entry per epoch with `train_acc` and per-split accuracy (retain / forget /
val / test) and `duration` — exactly like MUKSB. The driver **mirrors** it into
`results/<method>/seed<N>/` after each run, and `aggregate_results.py` folds it
(plus the final metrics) into `results/per_run/<method>_seed<N>.json`. (IU is a
one-shot method, so its `epochs` list is empty — only final metrics apply.)

## Run

```bash
conda activate salun          # env with torch used for this code
cd cifar10_random_forget

# all 9 methods, ONE seed (default = seed 1), each method on its own GPU (0-7)
nohup bash run_all_methods.sh > run_all.out 2>&1 &

# later: add more seeds (one GPU each)
SEEDS="1 2 3 4 5" nohup bash run_all_methods.sh > run_all.out 2>&1 &

# then aggregate (works with however many seeds finished)
python aggregate_results.py --root checkpoints --out results
```

### Pick which methods to run

Pass method names as **positional arguments** (or use the `METHODS` env var):

```bash
bash run_all_methods.sh MUKSB              # only MUKSB
bash run_all_methods.sh FT GA MUKSB        # a subset
bash run_all_methods.sh                    # all 9 methods
bash run_all_methods.sh --help             # usage + valid names
METHODS="FT MUKSB" bash run_all_methods.sh # same, via env var
```

Valid names: `retrain FT GA IU BE l1sparse SalUn MUNBa MUKSB` (unknown names are
rejected up front).

### Other overrides (env vars, space-separated)

```bash
SEEDS="1 2 3"   bash run_all_methods.sh MUKSB   # choose seeds
GPUS="6 7"      bash run_all_methods.sh         # round-robin seeds over these GPUs
SKIP_DONE=false bash run_all_methods.sh FT      # re-run already-finished jobs
```

**Parallelism:** methods run **in parallel, each on a different GPU** (round-robin
over `GPUS`, default `0 1 2 3 4 5 6 7`); seeds run **sequentially** (a seed's full
method-sweep finishes before the next seed starts). Restrict/choose GPUs with e.g.
`GPUS="4 5 1 7"`; if more methods than GPUs are requested, the extras wrap around
and share. Method names are case-insensitive. `SKIP_DONE=true` (default) skips any
`(method, seed)` whose `*eval_result.pth.tar` already exists, so it's resumable.

## Per-method hyperparameters

Edit the `CFG` table at the top of `run_all_methods.sh`. Format:
`<key>|<lr>|<epochs>|<batch>|<decreasing_lr>|<extra flags>`

| Method    | lr     | epochs | batch | extra                       | source           |
|-----------|--------|--------|-------|-----------------------------|------------------|
| retrain   | 0.1    | 160    | 256   | dec_lr 80,120               | repo convention  |
| FT        | 0.01   | 10     | 256   | —                           | benchmark default|
| GA        | 0.0001 | 5      | 256   | —                           | benchmark default|
| IU        | —      | 1*     | 256   | `--iu_damping 1e-3 --iu_scale 1.0` | benchmark default |
| BE        | 0.0001 | 10     | 256   | —                           | benchmark default|
| l1sparse  | 0.01   | 10     | 256   | `--with_l1 --alpha 5e-4`    | benchmark default|
| SalUn     | 0.01   | 10     | 256   | `--salun_density 0.5`       | benchmark default|
| MUNBa     | 0.03   | 10     | 256   | `--beta 1.0`                | repo (classwise) |
| MUKSB     | 0.03   | 10     | 256   | `--gamma 0.5 --alpha 0.2`   | repo (`run_pipeline.sh`) |

\* IU is a one-shot Newton step — `epochs` is ignored by the method.

> `retrain / MUKSB / MUNBa` values come from this repo's own scripts. The other
> baselines use standard unlearning-benchmark defaults — **review/tune them** for
> your setup before trusting the comparison.

## Metrics

`aggregate_results.py` reads each run's `evaluation_result` and reports
mean ± std over seeds:

- **Acc_Df** — accuracy on the forget set (lower is closer to retrain)
- **Acc_Dr** — accuracy on the retain set (higher is better)
- **Acc_Dt** — accuracy on the test set (higher is better)
- **MIA** — membership-inference accuracy, forgotten vs unseen (as %)
- **Avg.Gap** — mean |method − retrain| over the four metrics (lower is better)

Requires `arg_parser.py` to define `--salun_density / --iu_damping / --iu_scale`
(added alongside this folder).

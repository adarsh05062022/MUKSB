"""
Experiments/cost_comparison/compare_methods_flops.py
=====================================================
Computational-Cost Comparison Across NSFW Concept-Removal Methods
(MUKSB submission — unlearning-cost analysis)

Compares total training FLOPs for FIVE methods on the SAME SD-v1.4 UNet
backbone, so the comparison is architecture-matched and fair:

  (1) ESD          — train_scripts/train-esd-nsfw.py
  (2) SalUn        — train_scripts/random_label.py + generate_mask.py (one-time)
  (3) MUNBa        — Nash-bargaining gradient merge (no mask)
  (4) AdvUnlearn   — train-scripts/AdvUnlearn.py  (adversarial prompt attack +
                     reg retain), trains the CLIP text encoder
  (5) MUKSB (Ours) — MUKSB_nsfw.py  (magnitude-aware KS bargaining, no mask)

Of these, ONLY SalUn uses a parameter mask. MUKSB is run in its default
configuration (mask_variant=None) so, like MUNBa/ESD/AdvUnlearn, it updates
its full trainable parameter set — there is no mask-build overhead to charge.

Methodology  (same convention as Classification/compute_analysis.py + Table 9)
-----------
1. Load the SD-v1.4 UNet ONCE.
2. Measure one UNet forward pass on a single 512x512 latent (4x64x64) with
   fvcore (= F_fwd, the per-sample cost). Backward = 2 x forward.
3. For each method, count the TOTAL number of UNet sample-passes over the whole
   run (training + any one-time setup). A batched op of size B counts as B
   sample-passes, i.e. BATCH SIZE IS FOLDED IN — this is the real hardware cost:
       total_FLOPs = total_fwd_passes * F_fwd + total_bwd_passes * F_bwd
4. Write a comparison table (txt + csv + json) and a bar chart for the paper.

Why this is fair
----------------
* Forward FLOPs use the same UNet for every method (matched architecture).
* Backward = 2x forward is the standard accounting.
* DDIM-sampling FLOPs (the dominant term for ESD and AdvUnlearn) are counted
  explicitly — each quick_sample_till_t call averages `ddim_steps` UNet passes
  with classifier-free guidance (scale != 1 doubles the batch).
* SalUn's one-time saliency mask-build cost is INCLUDED.
* Batch size is folded into the pass counts per method (MUKSB/MUNBa/SalUn use
  batch=8; ESD batch=1; AdvUnlearn unlearn batch=1, retain batch=5), so the
  numbers are true hardware FLOPs comparable to the CIFAR-10 Table 9 totals.
* Pure O(P) arithmetic (Adam update, KS/Nash gradient merge) is OMITTED (~0.1%
  of one UNet pass), exactly as in the classification analysis. MUKSB and MUNBa
  therefore differ ONLY by MUNBa's extra Nash backward (3 bwd vs MUKSB's 2) —
  the same distinction reported in Table 9 (MUKSB 112.1 < MUNBa 119.7 TFLOPs).

Defaults match the reported NSFW runs:
  MUKSB/MUNBa/SalUn : batch=8, epochs=1, |F|=|R|=800            -> 100 steps
  ESD               : 1000 iters, ddim_steps=50
  AdvUnlearn        : 1000 iters, warmup=200, attack_step=30,
                      adv_prompt_update_step=1, retain_train=reg, ddim_steps=50

Usage (from SD/):
  conda activate munba3
  python Experiments/cost_comparison/compare_methods_flops.py --device 0
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

import torch

# -- path setup ---------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SD_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))   # .../SD
if _SD_DIR not in sys.path:
    sys.path.insert(0, _SD_DIR)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

# SD-v1.4 uses the CLIP ViT-L/14 text encoder; AdvUnlearn trains it in full.
CLIP_TEXTENCODER_PARAMS = 123_060_480


def load_unet(config_path: str, device: str):
    """
    Instantiate ONLY the SD UNet from the inference config.

    We do not need the autoencoder / CLIP / checkpoint weights to count FLOPs
    (a random-init UNet has identical architecture, params and per-pass FLOPs),
    so we skip the full LatentDiffusion load — which also avoids the heavy
    `taming` dependency and the 7 GB ckpt. Gradient checkpointing is disabled
    and the `checkpoint` helper is monkeypatched to a pass-through so fvcore can
    trace a clean forward graph.
    """
    import ldm.modules.diffusionmodules.util as _duu
    import ldm.modules.attention as _attn
    import ldm.modules.diffusionmodules.openaimodel as _oam

    def _passthrough(func, inputs, params, flag):
        return func(*inputs)
    _duu.checkpoint = _attn.checkpoint = _oam.checkpoint = _passthrough

    cfg = OmegaConf.load(config_path)
    unet_cfg = cfg.model.params.unet_config
    if "params" in unet_cfg:
        unet_cfg.params.use_checkpoint = False
    unet = instantiate_from_config(unet_cfg).to(device).eval()
    return unet


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def fmt(flops: float) -> str:
    if flops >= 1e18: return f"{flops/1e18:.3f} EFLOPs"
    if flops >= 1e15: return f"{flops/1e15:.3f} PFLOPs"
    if flops >= 1e12: return f"{flops/1e12:.3f} TFLOPs"
    if flops >= 1e9:  return f"{flops/1e9:.3f} GFLOPs"
    if flops >= 1e6:  return f"{flops/1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


# -----------------------------------------------------------------------------
# Method cost spec
# -----------------------------------------------------------------------------

@dataclass
class MethodCost:
    name: str
    # TOTAL UNet pass counts over the whole run (training only)
    train_fwd:   float
    train_bwd:   float
    train_extra: float = 0.0          # Nash/KS merge + Adam, summed over steps
    # one-time set-up (e.g. SalUn saliency mask)
    setup_fwd:   float = 0.0
    setup_bwd:   float = 0.0
    setup_extra: float = 0.0
    # bookkeeping / notes
    steps:       int   = 0
    fwd_note:    str   = ""
    extra_note:  str   = ""
    setup_note:  str   = ""
    # computed
    train_flops: float = 0.0
    setup_flops: float = 0.0
    total_flops: float = 0.0
    per_step_flops: float = 0.0

    def compute(self, F_fwd: float, F_bwd: float):
        self.train_flops = self.train_fwd * F_fwd + self.train_bwd * F_bwd + self.train_extra
        self.setup_flops = self.setup_fwd * F_fwd + self.setup_bwd * F_bwd + self.setup_extra
        self.total_flops = self.train_flops + self.setup_flops
        self.per_step_flops = self.train_flops / max(1, self.steps)


# -----------------------------------------------------------------------------
# Build the cost specs from the training configs
# -----------------------------------------------------------------------------

def build_methods(args, n_unet_params: int, n_textenc_params: int) -> List[MethodCost]:
    """
    Construct MethodCost objects for the 5 methods given the user's training
    configuration. All pass counts are derived from the actual training loops.

    Accounting convention (matches Classification/compute_analysis.py and Table 9):
      * The unit is one UNet pass on a SINGLE 512x512 latent (F_fwd); a batched
        op of size B counts as B such sample-passes, i.e. the per-step pass
        counts are MULTIPLIED BY THE BATCH SIZE. This is the true hardware cost.
      * backward = 2 x forward.
      * Pure O(P) arithmetic (Adam update, KS/Nash gradient merge) is NOT charged
        - it is ~0.1% of a single UNet pass and is omitted, exactly as in the
        classification analysis. The MUKSB-vs-MUNBa cost gap therefore comes
        ONLY from the number of UNet passes (see below), not from the merge.

    Per-step pass counts (read from the training code):
      ESD        : (ddim + 3) fwd + 1 bwd, batch 1            [train-esd-nsfw.py]
      SalUn      : 3 fwd + 1 bwd, batch B  + one-time mask     [random_label.py +
                                                                generate_mask.py]
      MUNBa      : 3 fwd + 3 bwd, batch B  (2 autograd.grad + a 3rd backward on
                   the Nash-weighted sum  -> the extra backward MUKSB avoids)
      MUKSB      : 3 fwd + 2 bwd, batch B  (2 autograd.grad; KS merge is pure
                   arithmetic, NO 3rd backward)               [MUKSB_nsfw.py]
      AdvUnlearn : DDIM-sampling reg loss + per-iter PGD attack [AdvUnlearn.py]
    """
    B          = args.batch_size                                    # 8
    grad_steps = args.epochs * max(1, args.forget_size // B)        # 1 * (800//8) = 100
    ddim       = args.ddim_steps                                    # avg UNet passes / sample (CFG)

    methods: List[MethodCost] = []

    # -- ESD (batch 1) ----------------------------------------------------------
    # per iter: z = quick_sample_till_t (~ddim CFG passes) + e_0,e_p,e_n (3 fwd)
    #           + loss.backward() (1 bwd).
    esd_fwd_ps = ddim + 3
    methods.append(MethodCost(
        name      = "ESD",
        train_fwd = esd_fwd_ps * 1 * args.esd_iters,
        train_bwd = 1 * 1 * args.esd_iters,
        steps     = args.esd_iters,
        fwd_note  = f"batch 1: {ddim}(DDIM,CFG) + 3 (e_0,e_p,e_n) fwd + 1 bwd, x{args.esd_iters} iters",
    ))

    # -- SalUn = Random-Label unlearning + saliency mask ------------------------
    # random_label.py per step (batch B): remain shared_step (1 fwd) + forget
    #   apply_model (1 fwd) + pseudo apply_model.detach() (1 fwd) + combined
    #   backward (1 bwd).
    # generate_mask.py (NSFW, one-time): per batch (bs_mask) 1 fwd + 1 bwd over
    #   ceil(|F|/bs_mask) batches  ==  one pass over the forget set.
    salun_mask_batches = max(1, args.forget_size // args.salun_mask_batch_size)
    methods.append(MethodCost(
        name      = "SalUn",
        train_fwd = 3 * B * grad_steps,
        train_bwd = 1 * B * grad_steps,
        setup_fwd = 1 * args.salun_mask_batch_size * salun_mask_batches,
        setup_bwd = 1 * args.salun_mask_batch_size * salun_mask_batches,
        steps     = grad_steps,
        fwd_note  = f"batch {B}: 3 fwd (remain, forget, pseudo) + 1 bwd per step",
        extra_note= f"the ONLY method with a mask (rho={args.mask_density:.0%}); merge/Adam O(P) omitted",
        setup_note= (f"saliency mask: one pass over |F|={args.forget_size} "
                     f"({salun_mask_batches} batches x bs {args.salun_mask_batch_size}, 1 fwd + 1 bwd)"),
    ))

    # -- MUNBa (Nash bargaining, no mask) ---------------------------------------
    # per step (batch B): remain shared_step + forget + pseudo (3 fwd);
    #   autograd.grad(loss_r) + autograd.grad(loss_u) (2 bwd) + a final
    #   loss.backward() on the Nash-weighted sum (3rd bwd).
    methods.append(MethodCost(
        name      = "MUNBa",
        train_fwd = 3 * B * grad_steps,
        train_bwd = 3 * B * grad_steps,
        steps     = grad_steps,
        fwd_note  = f"batch {B}: 3 fwd + 3 bwd (grads_r, grads_f, + Nash-weighted backward) per step",
        extra_note= "extra 3rd backward for the Nash merge (vs MUKSB's 2); Nash/Adam O(P) omitted",
    ))

    # -- AdvUnlearn (adversarial prompt attack + reg retain) --------------------
    # train-scripts/AdvUnlearn.py, train_method='text_encoder_full',
    # retain_train='reg', attack_method='pgd'.  CLIP text encoder is the target.
    #
    # Main loss per iter (get_train_loss_retain 'reg', unlearn batch 1, retain batch R):
    #   z (b1) ~ddim   + e_0,e_p (b1, 2 fwd) + e_n (b1, 1 fwd)
    #   retain_z (bR) ~ddim*R + retain_e_p (bR, R) + retain_e_n (bR, R)
    #   loss.backward() over the grad graphs (e_n: 1 + retain_e_n: R sample-bwd)
    # Attack (soft_prompt_attack), iters i>=warmup with i % update_step == 0;
    #   attack_step PGD steps, each (batch 1): z ~ddim + e_0,e_p,e_n (3 fwd) + 1 bwd.
    R = args.adv_retain_batch
    n_attack_rounds = sum(
        1 for i in range(args.adv_iterations)
        if i >= args.adv_warmup_iter and (i % args.adv_prompt_update_step == 0)
    )
    adv_main_fwd = ((ddim + 3) * 1 + (ddim + 2) * R) * args.adv_iterations
    adv_main_bwd = (1 + R) * args.adv_iterations
    adv_atk_fwd  = n_attack_rounds * args.adv_attack_step * (ddim + 3) * 1
    adv_atk_bwd  = n_attack_rounds * args.adv_attack_step * 1 * 1
    methods.append(MethodCost(
        name      = "AdvUnlearn",
        train_fwd = adv_main_fwd + adv_atk_fwd,
        train_bwd = adv_main_bwd + adv_atk_bwd,
        steps     = args.adv_iterations,
        fwd_note  = (f"main(b1+retain b{R}): (2xDDIM + 5 apply_model) /iter; "
                     f"attack: {n_attack_rounds} rounds x {args.adv_attack_step} PGD x ({ddim}+3) fwd"),
        extra_note= "trains CLIP text encoder; soft-prompt/Adam O(P) omitted",
    ))

    # -- MUKSB (Ours) : magnitude-aware KS bargaining, mask_variant=None --------
    # MUKSB_nsfw.py default (batch B): remain shared_step + forget + pseudo (3 fwd);
    #   autograd.grad(loss_r) + autograd.grad(loss_u) (2 bwd); KS merge is pure
    #   arithmetic -> NO 3rd backward.  No mask.
    methods.append(MethodCost(
        name      = "MUKSB (Ours)",
        train_fwd = 3 * B * grad_steps,
        train_bwd = 2 * B * grad_steps,
        steps     = grad_steps,
        fwd_note  = f"batch {B}: 3 fwd + 2 bwd (grads_r, grads_f) per step",
        extra_note= "KS merge is pure arithmetic (NO 3rd backward); no mask; O(P) omitted",
    ))

    return methods


# -----------------------------------------------------------------------------
# Forward-pass FLOP measurement
# -----------------------------------------------------------------------------

def measure_forward_flops(unet, image_size: int, device: str) -> int:
    """One UNet forward pass FLOPs. Falls back to params*pixels*3 if no fvcore."""
    h = w = image_size // 8
    dummy_x   = torch.randn(1, 4, h, w, device=device)
    dummy_t   = torch.tensor([500], device=device).long()
    dummy_ctx = torch.randn(1, 77, 768, device=device)

    if FVCORE_AVAILABLE:
        try:
            counter = FlopCountAnalysis(unet, (dummy_x, dummy_t, dummy_ctx))
            counter.unsupported_ops_warnings(False)
            counter.uncalled_modules_warnings(False)
            total = counter.total()
            if total > 0:
                print(f"  [fvcore]      one UNet forward = {fmt(total)}")
                return int(total)
            print("  [fvcore]      returned 0 - falling back to theoretical")
        except Exception as e:
            print(f"  [fvcore error] {e} - falling back to theoretical")

    n_params = sum(p.numel() for p in unet.parameters())
    flops    = n_params * h * w * 3
    print(f"  [theoretical] {n_params:,} params x {h*w} pixels x 3 = {fmt(flops)}")
    return int(flops)


def count_unet_params(unet, train_method: str = "full") -> int:
    n = 0
    for name, p in unet.named_parameters():
        keep = False
        if   train_method == "full":     keep = True
        elif train_method == "xattn":    keep = "attn2" in name
        elif train_method == "selfattn": keep = "attn1" in name
        elif train_method == "noxattn":  keep = not (name.startswith("out.")
                                                     or "attn2" in name
                                                     or "time_embed" in name)
        elif train_method == "notime":   keep = not (name.startswith("out.")
                                                     or "time_embed" in name)
        if keep:
            n += p.numel()
    return n


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def write_text_table(methods: List[MethodCost], cfg: dict, out_path: str):
    W_NAME, W_NUM = 16, 14
    header = (f"{'Method':<{W_NAME}}{'fwd/step*':>{W_NUM}}{'bwd/step*':>{W_NUM}}"
              f"{'Per-step':>{W_NUM}}{'Setup':>{W_NUM}}{'Total':>{W_NUM}}")
    H = "=" * len(header)
    lines = [H, "COMPUTATIONAL-COST COMPARISON - NSFW Concept Removal", H, ""]
    for k, v in cfg.items():
        lines.append(f"  {k:<28}: {v}")
    lines += ["", H, header, "-" * len(header)]
    for m in methods:
        lines.append(
            f"{m.name:<{W_NAME}}"
            f"{m.train_fwd/max(1,m.steps):>{W_NUM}.1f}"
            f"{m.train_bwd/max(1,m.steps):>{W_NUM}.1f}"
            f"{fmt(m.per_step_flops):>{W_NUM}}"
            f"{fmt(m.setup_flops):>{W_NUM}}"
            f"{fmt(m.total_flops):>{W_NUM}}"
        )
    lines.append(H)
    lines.append("  * fwd/step, bwd/step = UNet sample-passes per step (batch folded in).")

    lines += ["", "PER-METHOD BREAKDOWN", "-" * 70]
    for m in methods:
        lines.append(f"  {m.name}:")
        lines.append(f"      passes : {m.fwd_note}")
        lines.append(f"      extra  : {m.extra_note or '-'}")
        if m.setup_flops > 0:
            lines.append(f"      setup  : {m.setup_note}  ({fmt(m.setup_flops)})")
        lines.append(f"      total  : {m.steps:,} steps  ->  "
                     f"train {fmt(m.train_flops)} + setup {fmt(m.setup_flops)} "
                     f"= {fmt(m.total_flops)}")

    # narrative
    ours = next(m for m in methods if m.name.startswith("MUKSB"))
    lines += ["", "KEY FINDINGS", "-" * 70]
    lines.append(f"  MUKSB total cost = {fmt(ours.total_flops)} "
                 f"({ours.steps:,} steps, mask-free).")
    for m in methods:
        if m is ours:
            continue
        ratio = m.total_flops / ours.total_flops if ours.total_flops else float('nan')
        rel = f"{ratio:.2f}x heavier than MUKSB" if ratio >= 1 else f"{1/ratio:.2f}x lighter than MUKSB"
        lines.append(f"    vs {m.name:<12}: {fmt(m.total_flops):>14}  ({rel})")
    lines.append("")
    lines.append("  * Only SalUn carries a one-time mask-build cost; ESD, MUNBa,")
    lines.append("    AdvUnlearn and MUKSB update their full trainable parameter set.")
    lines.append("  * ESD and AdvUnlearn are dominated by DDIM sampling chains")
    lines.append(f"    ({cfg['ddim_steps']} CFG passes per sample); AdvUnlearn additionally")
    lines.append("    runs a multi-step PGD attack every iteration after warmup.")
    lines.append("  * MUKSB vs MUNBa: identical 3-forward pattern, but MUKSB's KS merge is")
    lines.append("    pure arithmetic (2 backward) whereas MUNBa needs a 3rd backward on the")
    lines.append("    Nash-weighted sum -> MUKSB saves one full backward pass per step.")
    lines.append("    (Same KS<Nash gap as CIFAR-10 Table 9: MUKSB 112.1 < MUNBa 119.7 TFLOPs.)")
    lines.append(H)

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text + "\n")


def write_csv(methods: List[MethodCost], cfg: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Config"])
        for k, v in cfg.items():
            w.writerow([f"# {k}", v])
        w.writerow([])
        w.writerow(["method", "steps", "train_fwd_passes", "train_bwd_passes",
                    "train_extra_flops", "setup_flops", "per_step_flops",
                    "train_flops", "total_flops", "fwd_note", "extra_note", "setup_note"])
        for m in methods:
            w.writerow([m.name, m.steps, m.train_fwd, m.train_bwd, m.train_extra,
                        m.setup_flops, m.per_step_flops, m.train_flops, m.total_flops,
                        m.fwd_note, m.extra_note, m.setup_note])


def write_json(methods: List[MethodCost], cfg: dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "methods": [asdict(m) for m in methods]},
                  f, indent=2, default=str)


def make_bar_chart(methods: List[MethodCost], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"  [plot skipped - matplotlib not available: {e}]")
        return

    names = [m.name for m in methods]
    total = [m.total_flops / 1e15 for m in methods]      # PFLOPs
    colors = ["#888", "#888", "#cc6666", "#d98a3a", "#3a7fbf"][:len(methods)]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(names, total, color=colors)
    ax.set_ylabel("Total training cost (PFLOPs)")
    ax.set_title("Unlearning cost - NSFW concept removal")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, total):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved -> {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    device = f"cuda:{args.device}"
    print("=" * 70)
    print("Computational-Cost Comparison - NSFW Concept Removal (MUKSB)")
    print("=" * 70)
    for k in ("ckpt_path", "config_path", "image_size", "train_method",
              "batch_size", "epochs", "forget_size", "esd_iters",
              "adv_iterations", "adv_attack_step", "ddim_steps"):
        print(f"  {k:<16}: {getattr(args, k)}")
    if not FVCORE_AVAILABLE:
        print("  WARNING: fvcore not installed - using theoretical estimate "
              "(pip install fvcore for hardware-level counts)")

    print("\nLoading SD-v1.4 UNet ...")
    t0 = time.time()
    unet = load_unet(args.config_path, device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    n_unet    = count_unet_params(unet, args.train_method)
    n_textenc = CLIP_TEXTENCODER_PARAMS
    print(f"  UNet trainable params (train_method='{args.train_method}'): {n_unet:,}")
    print(f"  CLIP text-encoder params (AdvUnlearn target)             : {n_textenc:,}")

    print("\nMeasuring one UNet forward pass ...")
    F_fwd = measure_forward_flops(unet, args.image_size, device)
    F_bwd = 2 * F_fwd
    print(f"  forward  = {fmt(F_fwd)}")
    print(f"  backward = {fmt(F_bwd)}  (= 2 x forward)")

    methods = build_methods(args, n_unet, n_textenc)
    for m in methods:
        m.compute(F_fwd, F_bwd)

    cfg = {
        "image_size":       args.image_size,
        "batch_size":       args.batch_size,
        "epochs":           args.epochs,
        "forget_size":      args.forget_size,
        "remain_size":      args.remain_size,
        "grad_steps":       args.epochs * max(1, args.forget_size // args.batch_size),
        "esd_iterations":   args.esd_iters,
        "adv_iterations":   args.adv_iterations,
        "adv_warmup_iter":  args.adv_warmup_iter,
        "adv_attack_step":  args.adv_attack_step,
        "adv_prompt_update_step": args.adv_prompt_update_step,
        "adv_retain_batch": args.adv_retain_batch,
        "ddim_steps":       args.ddim_steps,
        "train_method":     args.train_method,
        "mask_density(SalUn)": args.mask_density,
        "unet_params":      n_unet,
        "textencoder_params": n_textenc,
        "fvcore_used":      FVCORE_AVAILABLE,
        "forward_FLOPs":    F_fwd,
        "backward_FLOPs":   F_bwd,
    }

    results_dir = os.path.join(_THIS_DIR, "results")
    figs_dir    = os.path.join(_THIS_DIR, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    write_text_table(methods, cfg, os.path.join(results_dir, "flops_comparison.txt"))
    write_csv      (methods, cfg, os.path.join(results_dir, "flops_comparison.csv"))
    write_json     (methods, cfg, os.path.join(results_dir, "flops_comparison.json"))
    make_bar_chart (methods,      os.path.join(figs_dir,   "flops_comparison.png"))

    print(f"\nAll outputs written under {_THIS_DIR}/")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NSFW unlearning-cost comparison (FLOPs)")
    p.add_argument("--device",       type=str,   default="0")
    p.add_argument("--ckpt_path",    type=str,   default="models/ldm/sd-v1-4-full-ema.ckpt")
    p.add_argument("--config_path",  type=str,   default="configs/stable-diffusion/v1-inference.yaml")
    p.add_argument("--image_size",   type=int,   default=512)
    p.add_argument("--train_method", type=str,   default="full",
                   choices=["full", "noxattn", "xattn", "selfattn", "notime"])
    p.add_argument("--mask_density", type=float, default=0.50,
                   help="rho for SalUn's saliency mask (fraction of params active)")

    # shared gradient-method schedule (MUKSB / MUNBa / SalUn)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--forget_size",  type=int,   default=800)
    p.add_argument("--remain_size",  type=int,   default=800)
    p.add_argument("--salun_mask_batch_size", type=int, default=8,
                   help="generate_mask.py default batch size")

    # ESD schedule
    p.add_argument("--esd_iters",    type=int,   default=1000)

    # AdvUnlearn schedule (README defaults)
    p.add_argument("--adv_iterations",        type=int, default=1000)
    p.add_argument("--adv_warmup_iter",       type=int, default=200)
    p.add_argument("--adv_attack_step",       type=int, default=30)
    p.add_argument("--adv_prompt_update_step",type=int, default=1)
    p.add_argument("--adv_retain_batch",      type=int, default=5,
                   help="AdvUnlearn retain_batch (README default)")

    p.add_argument("--ddim_steps",   type=int,   default=50)

    args = p.parse_args()
    main(args)

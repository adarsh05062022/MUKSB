"""
Computational Analysis — ResNet-18 on CIFAR-10 (32×32)
=======================================================
Validated against source code for every method in Classification/unlearn/.

Corrections vs. first draft
-----------------------------
1. RL_proximal  — CIFAR-10 path concatenates forget+retain into one dataset
                  → 1 fwd + 1 bwd per step (NOT 2 fwd + 1 bwd).
2. SalUn        — forget and retain are processed in separate sequential loops,
                  each 1 fwd + 1 bwd per batch (NOT 2 fwd + 1 bwd).
3. boundary_sh  — FGSM uses a frozen test_model copy (deepcopy):
                  FGSM step:    1 fwd(grad) + 1 bwd *w.r.t. input* (test_model)
                  Inference:    1 fwd(no_grad) on test_model for adv_label
                  Train step:   1 fwd(grad) + 1 bwd *w.r.t. weights* (trainable model)
                  → 2 fwd(grad) + 1 fwd(no_grad) + 2 bwd (NOT 2 fwd + 2 bwd).
4. Wfisher      — three phases: forget-grad (N_f batches × 1 bwd),
                  retain-grad (N_r batches × 1 bwd),
                  woodfisher iterative (up to 1000 single-sample bwd).
                  Total ≈ (N_r + N_f + 1000) × (fwd + bwd), NOT just 317 × (fwd+bwd).

Run:
  conda run -n munba python Classification/compute_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.models as tvm
from thop import profile

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1e12: return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:  return f"{n/1e9:.3f} GFLOPs"
    if n >= 1e6:  return f"{n/1e6:.3f} MFLOPs"
    return f"{n/1e3:.3f} KFLOPs"

def fmt_macs(n):
    if n >= 1e9:  return f"{n/1e9:.3f} GMACs"
    if n >= 1e6:  return f"{n/1e6:.3f} MMACs"
    return f"{n/1e3:.3f} KMACs"

# ─────────────────────────────────────────────────────────────────────────────
# 1. ResNet-18 MACs — batch_size=1, 32×32, 10 classes
#
# Settings: standard torchvision ResNet-18 (same BasicBlock [2,2,2,2] as the
# project's Classification/models/ResNet.py), input 3×32×32, num_classes=10.
# NOTE: thop reports per-sample MACs (batch_size=1). Actual training uses
#       batch_size=128, so multiply all step-FLOPs by 128 for real hardware cost.
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Building ResNet-18 (random init, CIFAR-10, 32×32) …")
model = tvm.resnet18(weights=None, num_classes=10).to(device).eval()
dummy_x = torch.randn(1, 3, 32, 32, device=device)

print("Counting MACs with thop …")
macs, params = profile(model, inputs=(dummy_x,), verbose=False)

fwd_flops = macs * 2          # 1 MAC = 2 FLOPs (mul + add)
bwd_flops = fwd_flops * 2     # backward ≈ 2× forward (standard approximation)
step_1fwd1bwd = fwd_flops + bwd_flops   # cheapest possible step

print(f"\n{'='*65}")
print(f"  ResNet-18  (32×32 input, CIFAR-10, batch_size=1)")
print(f"{'='*65}")
print(f"  Architecture  : torchvision ResNet-18, BasicBlock [2,2,2,2]")
print(f"  Input shape   : [1, 3, 32, 32]")
print(f"  Parameters    : {params/1e6:.3f} M")
print(f"  MACs (fwd)    : {fmt_macs(macs)}")
print(f"  FLOPs (fwd)   : {fmt(fwd_flops)}")
print(f"  FLOPs (bwd)   : {fmt(bwd_flops)}  (≈ 2× forward; standard rule-of-thumb)")
print(f"  1 fwd+1 bwd   : {fmt(step_1fwd1bwd)}  ← cheapest-method baseline")
print(f"\n  ⚠  All FLOPs below are per-sample (batch_size=1).")
print(f"     Multiply by actual batch size (typically 128) for real cost.")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ITERATIVE METHODS — per-optimisation-step cost
#
# Source-validated breakdown for each method (CIFAR-10 path):
#
#  GA            — GA.py:
#    forget loader only. Loss = -CE(forget). 1 fwd + 1 bwd.
#
#  FT            — FT.py (FT_iter, non-imagenet path):
#    retain loader only. Loss = CE(retain). 1 fwd + 1 bwd.
#    (model called with is_feature=1 to return features, same FLOPs)
#
#  RL            — RL.py (CIFAR-10 path):
#    forget_dataset.targets randomised → concatenated with retain into one
#    DataLoader → single forward + backward on merged batch.
#    1 fwd + 1 bwd. Same structure as GA/FT.
#
#  RL_proximal   — RL_pro.py (CIFAR-10 path):
#    Same as RL: randomised forget + retain concatenated → 1 fwd + 1 bwd.
#    After optimizer.step(), a proximal clipping step runs in O(P) arithmetic
#    (no gradient involved). Per-step cost = 1 fwd + 1 bwd.
#    [SVHN path processes forget and retain in separate loops — still
#     1 fwd + 1 bwd per batch, just alternating.]
#
#  SalUn         — SalUn.py:
#    One-time mask phase: 1 fwd + 1 bwd per forget batch (see below).
#    Main loop: TWO separate sequential for-loops (forget, then retain),
#    each doing 1 fwd + 1 bwd + optimizer.step() independently.
#    Per-batch cost = 1 fwd + 1 bwd (NOT a joint 2-fwd step).
#
#  MUNBa         — MUNBa.py (both-loaders branch):
#    model(image_r)  → fwd 1 (retain)
#    model(image_u)  → fwd 2 (forget, with random label)
#    autograd.grad(loss_r, params, retain_graph=True)  → bwd 1
#    autograd.grad(loss_u, params, retain_graph=True)  → bwd 2
#    loss.backward() on Nash-weighted sum → bwd 3
#    Total: 2 fwd(grad) + 0 fwd(no_grad) + 3 bwd.
#
#  MUKSB (ours)  — MUKSB.py (both-loaders branch):
#    model(image_r) + autograd.grad(loss_r, retain_graph=False) → fwd1 + bwd1
#    model(image_u) + autograd.grad(loss_u, retain_graph=False) → fwd2 + bwd2
#    KS-bargaining merge (pure arithmetic) → no backward.
#    unpack_to_grads + optimizer.step() → no backward.
#    Total: 2 fwd(grad) + 0 fwd(no_grad) + 2 bwd.
#
#  SHs           — SHs.py (main per-step, both-loaders branch):
#    model(image_r) + model(image_u) computed, combined loss computed,
#    loss = loss_r - lam * loss_u → loss.backward() once.
#    Total: 2 fwd(grad) + 0 fwd(no_grad) + 1 bwd.
#    (epoch-0 SNIP pruning adds prune_num × 1 fwd + 1 bwd as one-time cost)
#    (with args.project=True: memory_num extra backward passes per epoch through
#     a proxy_model copy — not counted in per-step cost)
#
#  boundary_ex   — boundary_ex.py:
#    forget loader only; target relabelled to num_classes (new class).
#    1 fwd + 1 bwd.
#    Pre-step: expand_model() — modifies fc layer in-place, O(P) arithmetic.
#
#  boundary_sh   — boundary_sh.py:
#    test_model = deepcopy(model)  ← frozen copy, doubles GPU memory.
#    FGSM_perturb(x, y, model=test_model):
#      x_adv.requires_grad=True; test_model(x_adv) → fwd1 (w.r.t. input grad)
#      loss.backward() → bwd1 (input gradient ∂loss/∂x_adv, NOT ∂loss/∂params)
#    test_model(image_adv) → adv_label  → fwd2 (no_grad, inference only)
#    model(image) + loss.backward() on adv_label → fwd3(grad) + bwd2 (∂/∂params)
#    Total: 2 fwd(grad) + 1 fwd(no_grad) + 2 bwd.
#
#  Retrain       — retrain.py:
#    Standard CE on retain set. 1 fwd + 1 bwd.
# ─────────────────────────────────────────────────────────────────────────────

iterative = [
    # name,               fwd_g, fwd_ng, n_bwd, data_used,          source_note
    ("GA",                    1,      0,     1,  "forget",           "−CE(forget); gradient ascent"),
    ("FT",                    1,      0,     1,  "retain",           "CE(retain); standard fine-tune"),
    ("RL",                    1,      0,     1,  "forget+retain",    "random-label CE on merged dataset"),
    ("RL_proximal",           1,      0,     1,  "forget+retain",    "same as RL + O(P) proximal clipping (no bwd)"),
    ("SalUn",                 1,      0,     1,  "forget / retain",  "sequential loops, each 1fwd+1bwd; one-time mask phase extra"),
    ("Retrain",               1,      0,     1,  "retain",           "CE(retain); full re-train from scratch"),
    ("boundary_ex",           1,      0,     1,  "forget",           "CE(forget → new class); expand fc first (no bwd)"),
    ("SHs",                   2,      0,     1,  "forget+retain",    "combined loss = CE(retain)−λ·CE(forget); 1 bwd"),
    ("MUKSB (ours)",          2,      0,     2,  "forget+retain",    "2 separate autograd.grad; KS merge (no bwd)"),
    ("boundary_sh",           2,      1,     2,  "forget",           "FGSM on test_model(input grad)+train step; deepcopy=2× mem"),
    ("MUNBa",                 2,      0,     3,  "forget+retain",    "2×autograd.grad(retain_graph=True) + 1 final bwd"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. ONE-SHOT / CLOSED-FORM methods — total unlearning cost
#
# Dataset sizes (standard CIFAR-10 class-level forget):
#   N_forget = 4500   (one class of 50k, 10%)
#   N_retain = 40500  (remaining 90%)
#   Batch B  = 128
#
# IU (IU.py):
#   Phase 1 — diagonal Fisher on retain:
#     For each retain batch: 1 batched fwd (B samples) + B per-sample bwd
#     (retain_graph=True for all but last per-sample bwd in each batch)
#     Total forward passes:  N_r  (1 per sample, amortised over batches)
#     Total backward passes: N_r  (1 per-sample scalar backward)
#   Phase 2 — forget gradient:
#     For each forget batch: 1 batched fwd + 1 batched bwd
#     Total fwd: N_f,  Total bwd: N_f  (treating batched bwd as B single-sample bwds)
#   Phase 3 — Newton step: O(P) arithmetic, negligible.
#   Grand total: (N_r + N_f) fwd + (N_r + N_f) bwd
#
# Fisher (fisher.py):
#   fisher_information_matrix: same per-sample bwd as IU Phase 1.
#   Parameter update: O(P) noise addition, no backward.
#   Grand total: N_r fwd + N_r bwd
#
# Wfisher (Wfisher.py):
#   Phase 1 — forget grad (batch_size=args.batch_size):
#     N_f fwd + N_f bwd  (1 fwd+bwd per batch × B samples per batch)
#   Phase 2 — retain grad (batch_size=args.batch_size):
#     N_r fwd + N_r bwd
#   Phase 3 — woodfisher (batch_size=1, up to 1000 samples):
#     1000 fwd + 1000 bwd
#   Grand total: (N_r + N_f + 1000) × (fwd + bwd)
#
# SHs SNIP (one-time at epoch 0):
#   prune_num iterations on forget_loader: prune_num fwd + prune_num bwd
#   (default prune_num not fixed in code; typically O(10) iterations)
# ─────────────────────────────────────────────────────────────────────────────

N_r, N_f, B = 40500, 4500, 128

oneshot = [
    # name,       fwd_equiv,      bwd_equiv,         phases
    ("IU",        N_r + N_f,      N_r + N_f,
     f"Fisher: {N_r} per-sample bwd; Forget grad: {N_f} batched bwd equiv."),
    ("Fisher",    N_r,            N_r,
     f"Diagonal FIM: {N_r} per-sample bwd on retain set only"),
    ("Wfisher",   N_r+N_f+1000,  N_r+N_f+1000,
     f"Forget grad ({N_f}) + Retain grad ({N_r}) + Woodfisher ({1000} single-sample bwd)"),
]

# SalUn one-time mask generation
salun_mask_fwd = N_f // B          # one batched fwd per forget batch
salun_mask_bwd = N_f // B          # one batched bwd per forget batch
salun_mask_fwd_equiv = N_f         # equivalent single-sample passes
salun_mask_flops = (fwd_flops + bwd_flops) * salun_mask_fwd_equiv

# SHs one-time SNIP pruning (prune_num ≈ 10 typical)
PRUNE_NUM = 10
shs_snip_flops = (fwd_flops + bwd_flops) * PRUNE_NUM

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print results
# ─────────────────────────────────────────────────────────────────────────────

# ── Iterative table ──────────────────────────────────────────────────────────
baseline_flops = fwd_flops + bwd_flops  # cheapest: 1 fwd + 1 bwd

print(f"{'='*100}")
print(f"  Iterative methods — per optimisation step  (batch_size=1, input 32×32)")
print(f"  Sorted by increasing step cost")
print(f"{'='*100}")
print(f"  {'Method':<18} {'Fwd(grad)':>9} {'Fwd(∅grad)':>10} {'Bwd':>5}  {'Step FLOPs':<22}  {'Rel.':>5}  Data used")
print(f"  {'-'*97}")

# sort by step FLOPs
def step_flops(row):
    _, fg, fng, nb, _, _ = row
    return fwd_flops * (fg + fng) + bwd_flops * nb

for name, fwd_g, fwd_ng, n_bwd, data_used, note in sorted(iterative, key=step_flops):
    sf = fwd_flops * (fwd_g + fwd_ng) + bwd_flops * n_bwd
    rel = sf / baseline_flops
    print(f"  {name:<18} {fwd_g:>9d} {fwd_ng:>10d} {n_bwd:>5d}  {fmt(sf):<22}  {rel:>4.2f}x  {data_used}")

print(f"  {'-'*97}")
print(f"  Baseline (1 fwd+1 bwd) = {fmt(baseline_flops)}")
print(f"\n  Per-method notes:")
for name, fwd_g, fwd_ng, n_bwd, _, note in sorted(iterative, key=step_flops):
    print(f"    {name:<18}  {note}")

# ── One-time overheads ───────────────────────────────────────────────────────
print(f"\n  One-time overheads for iterative methods:")
print(f"    SalUn mask generation  : {N_f} forget samples → {fmt(salun_mask_flops)} extra")
print(f"    SHs SNIP pruning       : ~{PRUNE_NUM} forget batches → {fmt(shs_snip_flops)} extra (prune_num≈{PRUNE_NUM})")
print(f"    boundary_sh            : deepcopy of model in GPU memory (2× VRAM for weights)")

# ── One-shot table ───────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  One-shot / closed-form methods — TOTAL unlearning cost")
print(f"  N_retain={N_r}, N_forget={N_f}, batch_size={B}")
print(f"{'='*100}")
print(f"  {'Method':<12}  {'Fwd equiv.':>12}  {'Bwd equiv.':>12}  {'Total FLOPs':<24}  Note")
print(f"  {'-'*97}")
for name, fwd_eq, bwd_eq, note in oneshot:
    tf = fwd_flops * fwd_eq + bwd_flops * bwd_eq
    print(f"  {name:<12}  {fwd_eq:>12,}  {bwd_eq:>12,}  {fmt(tf):<24}  {note}")
print(f"  {'-'*97}")
print(f"  'equiv.' = equivalent single-sample passes. "
      f"Batched bwd counted as B single-sample bwd passes.")

# ── Memory notes ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Additional memory requirements (beyond base model)")
print(f"{'='*65}")
mem_notes = [
    ("GA/FT/RL/RL_prox/SalUn", "none"),
    ("Retrain/boundary_ex",     "none"),
    ("MUKSB (ours)",            "2 flat gradient buffers (gr_flat, gf_flat) ≈ 2×P floats"),
    ("MUNBa",                   "2 flat gradient buffers + retain_graph=True keeps full fwd graph in memory"),
    ("SHs (project=True)",      "proxy_model copy (1× model) + memory_num gradient vectors"),
    ("boundary_sh",             "test_model deepcopy (1× full model, e.g. ~43 MB for ResNet-18) in VRAM"),
    ("IU",                      "diagonal Fisher list ≈ P floats (same size as model parameters)"),
    ("Wfisher",                 "forget_grad + retain_grad + k_vec + o_vec ≈ 4×P floats"),
]
for name, note in mem_notes:
    print(f"  {name:<30}  {note}")

# ── LaTeX table ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  LaTeX table — iterative methods")
print(f"{'='*65}")

print(r"""\begin{table}[h]
\centering
\caption{Per-step computational cost for iterative machine-unlearning
         methods on ResNet-18 (CIFAR-10, $32\times32$, """ +
      f"{params/1e6:.1f}" + r"""M params,
         """ + fmt_macs(macs) + r""" per single-sample forward pass).
         Backward $\approx 2\times$ forward.
         All FLOPs are per-sample (batch\,=\,1); multiply by batch size for
         actual hardware cost.}
\label{tab:compute_cls}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccccl}
\toprule
Method & \makecell{Fwd\\(grad)} & \makecell{Fwd\\(no grad)} & Bwd
       & Step FLOPs & Data used \\
\midrule""")

for name, fwd_g, fwd_ng, n_bwd, data_used, _ in sorted(iterative, key=step_flops):
    sf = fwd_flops * (fwd_g + fwd_ng) + bwd_flops * n_bwd
    esc = name.replace("(ours)", r"\textbf{(ours)}")
    marker = r"$^\dagger$" if name == "SalUn" else \
             r"$^\ddagger$" if name == "SHs"  else \
             r"$^*$"        if name == "boundary_sh" else ""
    print(f"  {esc}{marker} & {fwd_g} & {fwd_ng} & {n_bwd} & "
          f"{fmt(fwd_flops*(fwd_g+fwd_ng)+bwd_flops*n_bwd)} & {data_used} \\\\")

print(r"""\midrule""")
for name, fwd_eq, bwd_eq, _ in oneshot:
    tf = fwd_flops * fwd_eq + bwd_flops * bwd_eq
    print(f"  {name} (one-shot) & — & — & {bwd_eq:,} & {fmt(tf)} & retain+forget \\\\")

print(r"""\bottomrule
\multicolumn{6}{l}{\footnotesize
  $^\dagger$ SalUn: additional one-time mask phase (""" + fmt(salun_mask_flops) + r""").} \\
\multicolumn{6}{l}{\footnotesize
  $^\ddagger$ SHs: additional one-time SNIP pruning (""" + fmt(shs_snip_flops) + r""", prune\_num$\approx$""" + str(PRUNE_NUM) + r""").} \\
\multicolumn{6}{l}{\footnotesize
  $^*$ boundary\_sh: requires a deepcopy of the model in GPU memory ($2\times$ VRAM for weights).}
\end{tabular}
\end{table}""")

# ── Settings summary ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Settings used for this analysis")
print(f"{'='*65}")
print(f"  Model          : torchvision ResNet-18 (BasicBlock [2,2,2,2])")
print(f"                   num_classes=10, random (untrained) weights")
print(f"  Input          : [1, 3, 32, 32]  (single sample, CIFAR-10)")
print(f"  FLOPs tool     : thop (v0.x) — counts multiply-accumulate ops")
print(f"  FLOPs formula  : fwd = 2×MACs; bwd ≈ 2×fwd (rule-of-thumb)")
print(f"  Dataset sizes  : N_retain={N_r:,}, N_forget={N_f:,}, batch={B}")
print(f"  Source files   : Classification/unlearn/{{GA,FT,RL,RL_pro,")
print(f"                   SalUn,MUNBa,MUKSB,SHs,boundary_ex,")
print(f"                   boundary_sh,retrain,IU,fisher,Wfisher}}.py")
print(f"\n  Caveats:")
print(f"  (a) 'bwd ≈ 2×fwd' is an approximation; real ratio depends on")
print(f"      layer types (attention layers can be closer to 1×).")
print(f"  (b) retain_graph=True (MUNBa, IU Fisher) keeps the computation")
print(f"      graph in GPU memory, increasing peak memory beyond what the")
print(f"      FLOPs count suggests.")
print(f"  (c) Methods that use is_feature=1 (FT, RL/SVHN path) return")
print(f"      intermediate features alongside logits; same FLOPs as standard.")
print(f"  (d) Batch-size scaling: all step-FLOPs scale linearly with batch.")
print(f"      At batch=128: multiply by 128.")
print(f"  (e) One-shot methods (IU, Fisher, Wfisher) have wall-clock time")
print(f"      dominated by the number of backward passes, not epochs.")

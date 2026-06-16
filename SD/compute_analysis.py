"""
Computational Analysis for SD Unlearning Methods
=================================================
Produces:
  1. MACs / FLOPs for one SD-UNet forward pass (via thop)
  2. Per-step backward-pass count for every method
  3. Estimated total FLOPs per optimisation step
  4. Parameter count for the UNet

Assumption: backward pass ≈ 2× forward FLOPs (standard rule-of-thumb).

Run:
  conda run -n munba python SD/compute_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from thop import profile, clever_format

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Instantiate the SD-1.x UNet (no weights needed — random init is enough)
# ─────────────────────────────────────────────────────────────────────────────
from ldm.modules.diffusionmodules.openaimodel import UNetModel

UNET_CFG = dict(
    image_size          = 32,      # latent spatial dim  (256-px image / 8× VAE)
    in_channels         = 4,
    out_channels        = 4,
    model_channels      = 320,
    attention_resolutions = [4, 2, 1],
    num_res_blocks      = 2,
    channel_mult        = (1, 2, 4, 4),
    num_heads           = 8,
    use_spatial_transformer = True,
    transformer_depth   = 1,
    context_dim         = 768,
    use_checkpoint      = False,
    legacy              = False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Building UNet (random weights) …")
unet = UNetModel(**UNET_CFG).to(device).eval()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Count MACs with thop
# ─────────────────────────────────────────────────────────────────────────────
BATCH   = 1
LAT_H   = 32   # latent height (256 / 8)
LAT_W   = 32   # latent width
SEQ_LEN = 77   # CLIP token length
EMB_DIM = 768

dummy_x   = torch.randn(BATCH, 4, LAT_H, LAT_W, device=device)
dummy_t   = torch.zeros(BATCH, dtype=torch.long, device=device)
dummy_ctx = torch.randn(BATCH, SEQ_LEN, EMB_DIM, device=device)

print("Counting MACs …")
macs, params = profile(unet, inputs=(dummy_x, dummy_t, dummy_ctx), verbose=False)

macs_str, params_str = clever_format([macs, params], "%.3f")

fwd_flops = macs * 2          # 1 MAC = 2 FLOPs (mul + add)
bwd_flops = fwd_flops * 2     # backward ≈ 2× forward (standard approximation)

fwd_str, bwd_str = clever_format([fwd_flops, bwd_flops], "%.3f")

print(f"\n{'='*60}")
print(f"  SD-1.x UNet — single forward pass")
print(f"{'='*60}")
print(f"  Parameters : {params_str}")
print(f"  MACs       : {macs_str}")
print(f"  FLOPs (fwd): {fwd_str}")
print(f"  FLOPs (bwd): {bwd_str}  (≈ 2× fwd)")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-method backward-pass table
# ─────────────────────────────────────────────────────────────────────────────
# Each entry:
#   (method_name,  n_fwd_in_graph,  n_fwd_no_grad,  n_bwd,  notes)
#
# FLOPs per step = fwd_flops * (n_fwd_in_graph + n_fwd_no_grad)
#                + bwd_flops * n_bwd
#
# Methods surveyed
#   MUKSB        — MUKSB_cls.py / MUKSB_nsfw.py
#   SalUn        — train_scripts/random_label.py
#   NSFW-removal — train_scripts/nsfw_removal.py  (same structure as SalUn)
#   ESD          — train_scripts/train-esd-nsfw.py

methods = [
    # name,               fwd_in_graph, fwd_no_grad, n_bwd, note
    ("MUKSB (ours)",            2,           0,       2,
     "autograd.grad(retain), autograd.grad(forget) — separate KS merge"),

    ("SalUn",                   2,           1,       1,
     "shared_step(retain)+apply_model(forget) in graph; pseudo.detach() no-grad"),

    ("NSFW-Removal",            2,           1,       1,
     "same structure as SalUn"),

    ("ESD",                     1,           2,       1,
     "frozen-model calls (no_grad×2); only trainable-model fwd in graph"),
]

# Additional one-time SalUn mask-generation phase
SALUN_MASK_BATCHES = 1000   # typical value; one backward per batch
salun_mask_extra   = SALUN_MASK_BATCHES * fwd_flops   # 1 fwd in graph per mask batch
# (not added per step; reported separately)

print(f"{'='*77}")
print(f"  Per-step Computational Cost  (batch size 1, latent 32×32)")
print(f"{'='*77}")
header = f"{'Method':<22} {'Fwd(grad)':>9} {'Fwd(∅grad)':>10} {'Bwd':>5} {'Step FLOPs':>14}  Notes"
print(header)
print(f"{'-'*77}")

def fmt_flops(n):
    if n >= 1e15:
        return f"{n/1e15:.3f} PFLOPs"
    if n >= 1e12:
        return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.3f} GFLOPs"
    return f"{n/1e6:.3f} MFLOPs"

for name, fwd_g, fwd_ng, n_bwd, note in methods:
    step_flops = fwd_flops * (fwd_g + fwd_ng) + bwd_flops * n_bwd
    step_str = fmt_flops(step_flops)
    print(f"  {name:<20} {fwd_g:>9d} {fwd_ng:>10d} {n_bwd:>5d}  {step_str:<20}  {note}")

print(f"{'-'*77}")

# relative cost vs cheapest method (ESD)
esd_flops = fwd_flops * (1 + 2) + bwd_flops * 1    # 3 fwd + 1 bwd

print(f"\n  Relative cost (1.0 = ESD baseline):")
for name, fwd_g, fwd_ng, n_bwd, _ in methods:
    step_flops = fwd_flops * (fwd_g + fwd_ng) + bwd_flops * n_bwd
    rel = step_flops / esd_flops
    print(f"    {name:<22}  {rel:.2f}x")

print(f"\n  SalUn mask-generation phase (one-time, {SALUN_MASK_BATCHES} batches):")
print(f"    Total extra FLOPs ≈ {fmt_flops(salun_mask_extra)}  (~{SALUN_MASK_BATCHES} fwd passes)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LaTeX table (ready to paste into paper)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  LaTeX table (copy-paste into paper)")
print(f"{'='*60}")

latex_header = r"""\begin{table}[h]
\centering
\caption{Computational cost per optimisation step for SD-1.x unlearning methods.
         Backbone: SD-1.x UNet (\textasciitilde{params} params, \textasciitilde{fwd_m}M MACs per forward pass).
         Backward pass $\approx 2\times$ forward FLOPs.}
\label{tab:compute}
\begin{tabular}{lcccc}
\toprule
Method & Fwd (grad) & Fwd (no grad) & Bwd & Step FLOPs \\
\midrule"""

fwd_m = int(macs / 1e6)
latex_header = latex_header.replace("{params}", params_str).replace("{fwd_m}", str(fwd_m))
print(latex_header)

for name, fwd_g, fwd_ng, n_bwd, _ in methods:
    step_flops = fwd_flops * (fwd_g + fwd_ng) + bwd_flops * n_bwd
    step_str = fmt_flops(step_flops)
    esc = name.replace("(ours)", r"\textbf{(ours)}")
    print(f"  {esc} & {fwd_g} & {fwd_ng} & {n_bwd} & {step_str} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

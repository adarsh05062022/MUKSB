"""
Evaluation/nsfw/generate_nsfw_multigpu.py
==========================================
Multi-GPU launcher for I2P NSFW generation.

Standalone usage:
    python generate_nsfw_multigpu.py --model_path models/my.pt --gpu_ids 0 1 2 3

Or called programmatically from eval_nsfw.py when --device receives multiple GPUs.

Worker mode (internal use, spawned by launcher):
    python generate_nsfw_multigpu.py --model_path ... --worker --gpu_id 0 --total_gpus 3 --gpu_ids 0 1 2
"""

import argparse
import gc
import os
import subprocess
import sys

import pandas as pd
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

I2P_CSV_DEFAULT = "/scratch/s25017/MUKSB/SD/prompts/coco_30k.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_row_slice(n_rows: int, partition_idx: int, total_partitions: int):
    """Return (start, end) row indices for this partition."""
    base  = n_rows // total_partitions
    rem   = n_rows %  total_partitions
    start = partition_idx * base
    end   = start + base + (rem if partition_idx == total_partitions - 1 else 0)
    return start, end


def load_pipeline(model_path: str, device: str):
    base         = "CompVis/stable-diffusion-v1-4"
    vae          = AutoencoderKL.from_pretrained(base, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet         = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        unet_state = {k.replace("model.diffusion_model.", ""): v
                      for k, v in state.items()
                      if k.startswith("model.diffusion_model.")}
        if unet_state:
            missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        else:
            missing, unexpected = unet.load_state_dict(state, strict=False)
        print(f"[UNet] loaded {model_path} | missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        print("[UNet] No checkpoint — using vanilla SD v1.4.")

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", num_train_timesteps=1000,
    )
    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.eval();     text_encoder.eval();     unet.eval()
    return vae, tokenizer, text_encoder, unet, scheduler


@torch.no_grad()
def generate_one(
    vae, tokenizer, text_encoder, unet, scheduler,
    prompt: str, seed: int, guidance_scale: float,
    image_size: int, ddim_steps: int, device: str,
) -> "Image.Image":
    text_input = tokenizer(
        [prompt], padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_emb   = text_encoder(text_input.input_ids.to(device))[0]
    uncond_inp = tokenizer(
        [""], padding="max_length",
        max_length=text_input.input_ids.shape[-1], return_tensors="pt",
    )
    uncond_emb = text_encoder(uncond_inp.input_ids.to(device))[0]
    cond_emb   = torch.cat([uncond_emb, text_emb])

    scheduler.set_timesteps(ddim_steps)
    generator = torch.manual_seed(seed)
    latents   = torch.randn(
        (1, unet.config.in_channels, image_size // 8, image_size // 8),
        generator=generator,
    ).to(device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        inp        = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = unet(inp, t, encoder_hidden_states=cond_emb).sample
        u, c       = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)
        latents    = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image   = vae.decode(latents).sample
    image   = (image / 2 + 0.5).clamp(0, 1)
    image   = image.cpu().permute(0, 2, 3, 1).numpy()
    image   = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


# ── Worker (called by subprocess) ────────────────────────────────────────────

def run_worker(args, partition_idx: int, total_partitions: int):
    """Run generation for this partition. Always uses cuda:0 because
    CUDA_VISIBLE_DEVICES is already set to the physical GPU by the launcher."""
    device    = "cuda:0"
    gpu_ids   = args.gpu_ids
    gpu_phys  = gpu_ids[partition_idx]
    model_tag = (os.path.basename(args.model_path).replace(".pt", "")
                 if args.model_path else "sd14_baseline")
    save_dir  = os.path.join(args.output_dir, model_tag)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(args.prompts_path)
    if "case_number" not in df.columns and df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "row_idx"})

    row_start, row_end = get_row_slice(len(df), partition_idx, total_partitions)
    df_slice = df.iloc[row_start:row_end].reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"Worker  gpu_ids[{partition_idx}] = cuda:{gpu_phys}  (visible as cuda:0)")
    print(f"Rows    [{row_start}, {row_end})  →  {len(df_slice)} prompts")
    print(f"Model   {model_tag}")
    print(f"Output  {save_dir}")
    print(f"{'='*65}\n")

    vae, tokenizer, text_encoder, unet, scheduler = load_pipeline(args.model_path, device)

    total_saved = 0
    for _, row in tqdm(df_slice.iterrows(), total=len(df_slice),
                       desc=f"GPU {gpu_phys}"):
        case_number  = int(row["case_number"])
        prompt       = str(row["prompt"])
        base_seed    = int(row["evaluation_seed"])
        img_guidance = float(row.get("evaluation_guidance", args.guidance_scale))

        for img_idx in range(args.n_per_prompt):
            out_path = os.path.join(save_dir, f"{case_number:05d}_{img_idx}.png")
            if os.path.exists(out_path):
                continue

            seed = base_seed + img_idx
            try:
                img = generate_one(
                    vae, tokenizer, text_encoder, unet, scheduler,
                    prompt=prompt, seed=seed, guidance_scale=img_guidance,
                    image_size=args.image_size, ddim_steps=args.ddim_steps,
                    device=device,
                )
                img.save(out_path)
                total_saved += 1
            except Exception as e:
                print(f"[WARN] GPU {gpu_phys}  case {case_number} img {img_idx} failed: {e}")

    del vae, text_encoder, unet, scheduler
    gc.collect(); torch.cuda.empty_cache()
    print(f"\n[GPU {gpu_phys}] Done — {total_saved} images written.")


# ── Launcher ─────────────────────────────────────────────────────────────────

def launch(args):
    """Spawn one subprocess per GPU in args.gpu_ids, wait for all to finish."""
    gpu_ids = args.gpu_ids
    total   = len(gpu_ids)
    df      = pd.read_csv(args.prompts_path)
    print(f"Launching {total} workers on GPUs: {gpu_ids}")
    print(f"Total prompts: {len(df)}  →  ~{len(df) // total} per GPU\n")

    log_dir = os.path.join(args.output_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for partition_idx, gpu_phys in enumerate(gpu_ids):
        cmd = [
            sys.executable, __file__,
            "--worker",
            "--gpu_id",        str(partition_idx),
            "--total_gpus",    str(total),
            "--gpu_ids",       *[str(g) for g in gpu_ids],
            "--model_path",    args.model_path,
            "--output_dir",    args.output_dir,
            "--prompts_path",  args.prompts_path,
            "--n_per_prompt",  str(args.n_per_prompt),
            "--guidance_scale", str(args.guidance_scale),
            "--image_size",    str(args.image_size),
            "--ddim_steps",    str(args.ddim_steps),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_phys)

        log_path = os.path.join(log_dir, f"worker_gpu{gpu_phys}.log")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        procs.append((proc, gpu_phys, log_path, log_file))
        print(f"  [launched] GPU {gpu_phys}  PID {proc.pid}  log → {log_path}")

    print("\nWaiting for all workers to finish …")
    failed = []
    for proc, gpu_phys, log_path, log_file in procs:
        proc.wait()
        log_file.close()
        if proc.returncode == 0:
            print(f"  GPU {gpu_phys}  [OK]  log: {log_path}")
        else:
            print(f"  GPU {gpu_phys}  [FAILED code={proc.returncode}]  log: {log_path}")
            failed.append(gpu_phys)

    if failed:
        raise RuntimeError(f"Workers failed on GPUs: {failed}")
    print("\nAll workers done.")


def launch_multigpu(
    model_path: str,
    output_dir: str,
    prompts_path: str,
    n_per_prompt: int,
    gpu_ids: list,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    ddim_steps: int = 50,
):
    """
    Callable entry-point for eval_nsfw.py.
    Blocks until all GPU workers finish.
    """
    import types
    args = types.SimpleNamespace(
        model_path    = model_path,
        output_dir    = output_dir,
        prompts_path  = prompts_path,
        n_per_prompt  = n_per_prompt,
        gpu_ids       = gpu_ids,
        guidance_scale= guidance_scale,
        image_size    = image_size,
        ddim_steps    = ddim_steps,
    )
    launch(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-GPU I2P NSFW generation launcher"
    )
    parser.add_argument("--model_path",     type=str, default="SD_baseline",
                        help="SSU .pt checkpoint (empty = SD v1.4 baseline)")
    parser.add_argument("--output_dir",     type=str,
                        default="Evaluation/nsfw/coco_30k")
    parser.add_argument("--prompts_path",   type=str, default=I2P_CSV_DEFAULT)
    parser.add_argument("--n_per_prompt",   type=int, default=1,
                        help="Number of images per prompt (default: 1)")
    parser.add_argument("--gpu_ids",        type=int, nargs="+", default=[0, 1, 2, 3, 5, 6, 7],
                        help="Physical GPU indices to use, e.g. --gpu_ids 0 1 2 3")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int, default=512)
    parser.add_argument("--ddim_steps",     type=int, default=50)

    # internal worker flags — do not set manually
    parser.add_argument("--worker",     action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id",     type=int, default=0,  help=argparse.SUPPRESS)
    parser.add_argument("--total_gpus", type=int, default=1,  help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        run_worker(args, partition_idx=args.gpu_id, total_partitions=args.total_gpus)
    else:
        launch(args)

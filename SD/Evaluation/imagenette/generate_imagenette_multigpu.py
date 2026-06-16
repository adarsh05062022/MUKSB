"""
Evaluation/imagenette/generate_imagenette_multigpu.py
======================================================
Multi-GPU launcher for Imagenette image generation.

Standalone usage:
    python generate_imagenette_multigpu.py --model_path models/my.pt --gpu_ids 0 1 2 3

Or called programmatically from eval_imagenette.py when --device receives multiple GPUs.

Worker mode (internal use, spawned by launcher):
    python generate_imagenette_multigpu.py --model_path ... --worker --gpu_id 0 --total_gpus 3 --gpu_ids 0 1 2
"""

import argparse
import gc
import os
import subprocess
import sys

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ── Imagenette class metadata ─────────────────────────────────────────────────
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
PROMPTS = [f"an image of a {c}" for c in IMAGENETTE_CLASSES]
CLASS_SEEDS = [4889, 4782, 4068, 4373, 987, 1562, 4264, 432, 1912, 1945]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gpu_slice(n_images: int, partition_idx: int, total_partitions: int):
    """Return (start, end) global image indices for this partition."""
    base  = n_images // total_partitions
    rem   = n_images %  total_partitions
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
        print(f"[UNet] loaded {model_path} | missing={len(missing)} unexpected={len(unexpected)}")
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
def generate_images_slice(
    vae, tokenizer, text_encoder, unet, scheduler,
    prompt, class_seed, global_start, global_end,
    device, guidance_scale, image_size, ddim_steps, save_dir,
    batch_size=4,
):
    os.makedirs(save_dir, exist_ok=True)
    existing     = {f for f in os.listdir(save_dir) if f.endswith(".png")}
    todo_indices = [i for i in range(global_start, global_end)
                    if f"{i:05d}.png" not in existing]
    if not todo_indices:
        print(f"  [SKIP] all {global_end - global_start} images already exist")
        return 0

    text_input = tokenizer([prompt], padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    text_emb   = text_encoder(text_input.input_ids.to(device))[0]
    uncond_inp = tokenizer([""], padding="max_length",
                           max_length=text_input.input_ids.shape[-1], return_tensors="pt")
    uncond_emb = text_encoder(uncond_inp.input_ids.to(device))[0]

    saved = 0
    pbar  = tqdm(total=len(todo_indices),
                 desc=f"  imgs {global_start}–{global_end-1}", leave=False)

    for batch_start in range(0, len(todo_indices), batch_size):
        batch_indices = todo_indices[batch_start: batch_start + batch_size]
        this_batch    = len(batch_indices)
        generator     = torch.manual_seed(class_seed + batch_indices[0])

        scheduler.set_timesteps(ddim_steps)

        latents = torch.randn(
            (this_batch, unet.config.in_channels, image_size // 8, image_size // 8),
            generator=generator,
        ).to(device) * scheduler.init_noise_sigma

        cond = torch.cat([uncond_emb.expand(this_batch, -1, -1),
                          text_emb.expand(this_batch, -1, -1)])

        for t in scheduler.timesteps:
            inp        = scheduler.scale_model_input(torch.cat([latents] * 2), t)
            noise_pred = unet(inp, t, encoder_hidden_states=cond).sample
            u, c       = noise_pred.chunk(2)
            noise_pred = u + guidance_scale * (c - u)
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

        images = vae.decode(1 / 0.18215 * latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")

        for img_arr, global_idx in zip(images, batch_indices):
            Image.fromarray(img_arr).save(os.path.join(save_dir, f"{global_idx:05d}.png"))
            saved += 1
            pbar.update(1)

    pbar.close()
    return saved


# ── Worker (called by subprocess) ────────────────────────────────────────────

def run_worker(args, partition_idx: int, total_partitions: int):
    """Run generation for this partition. Always uses cuda:0 because
    CUDA_VISIBLE_DEVICES is already set to the physical GPU by the launcher."""
    device       = "cuda:0"
    gpu_ids      = args.gpu_ids
    global_start, global_end = get_gpu_slice(args.n_images, partition_idx, total_partitions)
    n_local      = global_end - global_start
    model_tag    = os.path.basename(args.model_path).replace(".pt", "") if args.model_path else "sd14_baseline"
    out_root     = os.path.join(args.output_dir, model_tag)
    gpu_phys     = gpu_ids[partition_idx]

    print(f"\n{'='*65}")
    print(f"Worker  gpu_ids[{partition_idx}] = cuda:{gpu_phys}  (visible as cuda:0)")
    print(f"Slice   [{global_start}, {global_end})  →  {n_local} images/class")
    print(f"Model   {model_tag}")
    print(f"Output  {out_root}")
    print(f"{'='*65}\n")

    vae, tokenizer, text_encoder, unet, scheduler = load_pipeline(args.model_path, device)

    total_saved = 0
    for cls_idx, (cls_name, prompt, seed) in enumerate(
        zip(IMAGENETTE_CLASSES, PROMPTS, CLASS_SEEDS)
    ):
        save_dir = os.path.join(out_root, f"{cls_idx:02d}_{cls_name.replace(' ', '_')}")
        print(f"[GPU {gpu_phys}] class {cls_idx:2d}  {cls_name:<20}  '{prompt}'")
        n_saved = generate_images_slice(
            vae, tokenizer, text_encoder, unet, scheduler,
            prompt=prompt, class_seed=seed,
            global_start=global_start, global_end=global_end,
            device=device, guidance_scale=args.guidance_scale,
            image_size=args.image_size, ddim_steps=args.ddim_steps,
            save_dir=save_dir, batch_size=args.batch_size,
        )
        total_saved += n_saved
        print(f"         → saved {n_saved}  (running total: {total_saved})")

    del vae, text_encoder, unet, scheduler
    gc.collect(); torch.cuda.empty_cache()
    print(f"\n[GPU {gpu_phys}] Done — {total_saved} images written.")


# ── Launcher ─────────────────────────────────────────────────────────────────

def launch(args):
    """Spawn one subprocess per GPU in args.gpu_ids, wait for all to finish."""
    gpu_ids = args.gpu_ids
    total   = len(gpu_ids)
    print(f"Launching {total} workers on GPUs: {gpu_ids}")
    print(f"Total images/class: {args.n_images}  →  ~{args.n_images // total} per GPU\n")

    log_dir = os.path.join(args.output_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for partition_idx, gpu_phys in enumerate(gpu_ids):
        cmd = [
            sys.executable, __file__,
            "--worker",
            "--gpu_id",      str(partition_idx),
            "--total_gpus",  str(total),
            "--gpu_ids",     *[str(g) for g in gpu_ids],
            "--model_path",  args.model_path,
            "--output_dir",  args.output_dir,
            "--n_images",    str(args.n_images),
            "--guidance_scale", str(args.guidance_scale),
            "--image_size",  str(args.image_size),
            "--ddim_steps",  str(args.ddim_steps),
            "--batch_size",  str(args.batch_size),
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
    n_per_class: int,
    gpu_ids: list,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    ddim_steps: int = 50,
    batch_size: int = 4,
):
    """
    Callable entry-point for eval_imagenette.py.
    Blocks until all GPU workers finish.
    """
    import types
    args = types.SimpleNamespace(
        model_path    = model_path,
        output_dir    = output_dir,
        n_images      = n_per_class,
        gpu_ids       = gpu_ids,
        guidance_scale= guidance_scale,
        image_size    = image_size,
        ddim_steps    = ddim_steps,
        batch_size    = batch_size,
    )
    launch(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-GPU Imagenette image generation launcher"
    )
    parser.add_argument("--model_path",     type=str,
                        default="/scratch/s25017/MUKSB/SD/models/compvis-cls_0-MUKSB-g0.5-method_full-lr_1e-05_E3_U963_pseudo/diffusers-cls_0-MUKSB-g0.5-method_full-lr_1e-05_E3_U963_pseudo-epoch_1.pt",
                        help="SSU .pt checkpoint. Empty → SD v1.4 baseline.")
    parser.add_argument("--output_dir",     type=str,
                        default="Evaluation/imagenette/generated")
    parser.add_argument("--n_images",       type=int, default=500,
                        help="Total images per class (split across GPUs).")
    parser.add_argument("--gpu_ids",        type=int, nargs="+", default=[0],
                        help="Physical GPU indices to use, e.g. --gpu_ids 0 1 2 3")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size",     type=int, default=512)
    parser.add_argument("--ddim_steps",     type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=4)

    # internal worker flags — do not set manually
    parser.add_argument("--worker",     action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id",     type=int, default=0,  help=argparse.SUPPRESS)
    parser.add_argument("--total_gpus", type=int, default=1,  help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        run_worker(args, partition_idx=args.gpu_id, total_partitions=args.total_gpus)
    else:
        launch(args)

import copy
import logging
import os
import pickle
import random
import threading
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvu
import tqdm
from datasets import (
    all_but_one_class_path_dataset,
    data_transform,
    get_dataset,
    get_forget_dataset,
    inverse_data_transform,
)
from functions import create_class_labels, cycle, get_optimizer
from functions.denoising import generalized_steps_conditional
from functions.losses import loss_registry_conditional
from models.diffusion import Conditional_Model
from models.ema import EMAHelper
from runners.muksb_utils import (
    build_flat_mask,
    flatten_grads,
    ks_step,
    unpack_to_grads,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def save_fim(self):
        args, config = self.args, self.config
        bs = (
            torch.cuda.device_count()
        )  # process 1 sample per GPU, so bs == number of gpus
        fim_dataset = ImageFolder(
            os.path.join(args.ckpt_folder, "class_samples"),
            transform=transforms.ToTensor(),
        )
        fim_loader = DataLoader(
            fim_dataset,
            batch_size=bs,
            num_workers=config.data.num_workers,
            shuffle=True,
        )

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        model.eval()

        # calculate FIM
        fisher_dict = {}
        fisher_dict_temp_list = [{} for _ in range(bs)]

        for name, param in model.named_parameters():
            fisher_dict[name] = param.data.clone().zero_()

            for i in range(bs):
                fisher_dict_temp_list[i][name] = param.data.clone().zero_()

        # calculate Fisher information diagonals
        for step, data in enumerate(
            tqdm.tqdm(fim_loader, desc="Calculating Fisher information matrix")
        ):
            x, c = data
            x, c = x.to(self.device), c.to(self.device)

            b = self.betas
            ts = torch.chunk(torch.arange(0, self.num_timesteps), args.n_chunks)

            for _t in ts:
                for i in range(len(_t)):
                    e = torch.randn_like(x)
                    t = torch.tensor([_t[i]]).expand(bs).to(self.device)

                    # keepdim=True will return loss of shape [bs], so gradients across batch are NOT averaged yet
                    if i == 0:
                        loss = loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )
                    else:
                        loss += loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )

                # store first-order gradients for each sample separately in temp dictionary
                # for each timestep chunk
                for i in range(bs):
                    model.zero_grad()
                    if i != len(loss) - 1:
                        loss[i].backward(retain_graph=True)
                    else:
                        loss[i].backward()
                    for name, param in model.named_parameters():
                        fisher_dict_temp_list[i][name] += param.grad.data
                del loss

            # after looping through all 1000 time steps, we can now aggregrate each individual sample's gradient and square and average
            for name, param in model.named_parameters():
                for i in range(bs):
                    fisher_dict[name].data += (
                        fisher_dict_temp_list[i][name].data ** 2
                    ) / len(fim_loader.dataset)
                    fisher_dict_temp_list[i][name] = (
                        fisher_dict_temp_list[i][name].clone().zero_()
                    )

            if (step + 1) % config.training.save_freq == 0:
                with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
                    pickle.dump(fisher_dict, f)

        # save at the end
        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
            pickle.dump(fisher_dict, f)


    def train(self):
            args, config = self.args, self.config

            # DDP setup — falls back to DataParallel if not launched with torchrun
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            use_ddp = local_rank >= 0
            if use_ddp:
                torch.distributed.init_process_group(backend="nccl")
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(self.device)
                self.betas = self.betas.to(self.device)
            is_main = (not use_ddp) or (local_rank == 0)

            D_train_loader = get_dataset(args, config)
            D_train_iter = cycle(D_train_loader)

            model = Conditional_Model(config)
            model.to(self.device)

            if use_ddp:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank]
                )
            else:
                model = torch.nn.DataParallel(model)

            optimizer = get_optimizer(self.config, model.parameters())

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
            else:
                ema_helper = None

            # mixed precision scaler (BF16 on H100/H200, FP16 otherwise)
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

            model.train()

            start = time.time()
            for step in range(0, self.config.training.n_iters):

                x, c = next(D_train_iter)
                n = x.size(0)
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)

                if is_main and (step + 1) % self.config.training.log_freq == 0:
                    end = time.time()
                    logging.info(
                        f"step: {step}, loss: {loss.item():.4f}, time: {end-start:.2f}s"
                    )
                    start = time.time()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                scaler.step(optimizer)
                scaler.update()

                if self.config.model.ema:
                    ema_helper.update(model)

                if is_main and (step + 1) % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    ckpt_path = os.path.join(self.config.ckpt_dir, "ckpt.pth")

                    def _to_cpu(s):
                        if not isinstance(s, dict):
                            return s
                        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in s.items()}

                    save_states = [_to_cpu(s) for s in states]
                    threading.Thread(
                        target=torch.save, args=(save_states, ckpt_path), daemon=True
                    ).start()
                    logging.info(f"Checkpoint saved at step {step}")

                vis_freq = getattr(self.config.training, "vis_freq", self.config.training.snapshot_freq)
                if is_main and (step + 1) % vis_freq == 0:
                    test_model = ema_helper.ema_copy(model) if self.config.model.ema else copy.deepcopy(model)
                    test_model.eval()
                    self.sample_visualization(test_model, step, args.cond_scale)
                    del test_model

            if use_ddp:
                torch.distributed.destroy_process_group()


    def train_forget(self):
        args, config = self.args, self.config
        logging.info(
            f"Training diffusion forget with contrastive and EWC. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )
        D_train_loader = all_but_one_class_path_dataset(
            config,
            os.path.join(args.ckpt_folder, "class_samples"),
            args.label_to_forget,
        )
        D_train_iter = cycle(D_train_loader)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "rb") as f:
            fisher_dict = pickle.load(f)

        params_mle_dict = {}
        for name, param in model.named_parameters():
            params_mle_dict[name] = param.data.clone()

        label_choices = list(range(config.data.n_classes))
        label_choices.remove(args.label_to_forget)

        for step in range(0, config.training.n_iters):
            model.train()
            x_remember, c_remember = next(D_train_iter)
            x_remember, c_remember = x_remember.to(self.device), c_remember.to(
                self.device
            )
            x_remember = data_transform(config, x_remember)

            n = x_remember.size(0)
            channels = config.data.channels
            img_size = config.data.image_size
            c_forget = (torch.ones(n, dtype=int) * args.label_to_forget).to(self.device)
            x_forget = (
                torch.rand((n, channels, img_size, img_size), device=self.device) - 0.5
            ) * 2.0
            e_remember = torch.randn_like(x_remember)
            e_forget = torch.randn_like(x_forget)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](
                model, x_forget, t, c_forget, e_forget, b, cond_drop_prob=0.0
            ) + config.training.gamma * loss_registry_conditional[config.model.type](
                model, x_remember, t, c_remember, e_remember, b, cond_drop_prob=0.0
            )
            forgetting_loss = loss.item()

            ewc_loss = 0.0
            for name, param in model.named_parameters():
                _loss = (
                    fisher_dict[name].to(self.device)
                    * (param - params_mle_dict[name].to(self.device)) ** 2
                )
                loss += config.training.lmbda * _loss.sum()
                ewc_loss += config.training.lmbda * _loss.sum()

            if (step + 1) % config.training.log_freq == 0:
                logging.info(
                    f"step: {step}, loss: {loss.item()}, forgetting loss: {forgetting_loss}, ewc loss: {ewc_loss}"
                )

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    # epoch,
                    step,
                ]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model


    def retrain(self):
        args, config = self.args, self.config

        D_remain_loader, _ = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)

        model = Conditional_Model(config)

        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        model.train()

        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            x, c = next(D_remain_iter)
            # x, c = next(D_train_iter)

            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end-start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def saliency_unlearn(self):
        args, config = self.args, self.config

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        if args.mask_path:
            mask = torch.load(args.mask_path)
        else:
            mask = None

        print("Loading checkpoints {}".format(args.ckpt_folder))

        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())
        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        model.train()
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()

            # remain stage
            remain_x, remain_c = next(D_remain_iter)
            n = remain_x.size(0)
            remain_x = remain_x.to(self.device)
            remain_x = data_transform(self.config, remain_x)
            e = torch.randn_like(remain_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            remain_loss = loss_registry_conditional[config.model.type](
                model, remain_x, t, remain_c, e, b
            )

            # forget stage
            forget_x, forget_c = next(D_forget_iter)

            n = forget_x.size(0)
            forget_x = forget_x.to(self.device)
            forget_x = data_transform(self.config, forget_x)
            e = torch.randn_like(forget_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t, forget_c, e, b
                )

            else:
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                output = model(forget_x, t.float(), forget_c, mode="train")

                if args.method == "rl":
                    pseudo_c = torch.full(
                        forget_c.shape,
                        (args.label_to_forget + 1) % 10,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)

            loss = forget_loss + args.alpha * remain_loss

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end-start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name].to(param.grad.device)
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def muksb_unlearn(self):
        """
        MUKSB unlearning for conditional DDPM.

        Replaces the additive ``forget_loss + alpha * remain_loss`` objective
        from ``saliency_unlearn`` with a Kalai-Smorodinsky bargaining merge of
        the two gradients. ``args.gamma`` controls retain priority:

            gamma = 0.5  -> symmetric KS (default)
            gamma > 0.5  -> retain-favoured bisector (recommended for 1-of-10)
            gamma < 0.5  -> forget-favoured bisector

        Forget-loss variants use the existing ``--method`` flag:
            ga -> gradient ascent on the forget batch
            rl -> random-label (matches noise prediction under a pseudo class)

        ``--mask_path`` is honoured: KS bargaining is restricted to the masked
        coordinates, and the unmasked coordinates receive zero update.
        """
        args, config = self.args, self.config

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        if args.mask_path:
            mask = torch.load(args.mask_path, map_location="cpu")
        else:
            mask = None

        print("Loading checkpoints {}".format(args.ckpt_folder))

        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)

        # Single-GPU fast path: skip DataParallel overhead when only one CUDA
        # device is visible. DataParallel adds scatter/gather hooks that are
        # pure overhead with one device — and we save its state_dict elsewhere
        # with a "module." key prefix that we transparently strip here.
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_dp = n_gpus > 1
        if use_dp:
            model = torch.nn.DataParallel(model)

        def _strip_module(state):
            if any(k.startswith("module.") for k in state.keys()):
                return {
                    (k[len("module."):] if k.startswith("module.") else k): v
                    for k, v in state.items()
                }
            return state

        model_state = states[0] if use_dp else _strip_module(states[0])
        model.load_state_dict(model_state, strict=True)

        optimizer = get_optimizer(config, model.parameters())
        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            try:
                ema_state = states[-1] if use_dp else _strip_module(states[-1])
                ema_helper.load_state_dict(ema_state)
            except Exception:
                pass
        else:
            ema_helper = None

        # ── KS hyper-parameters ───────────────────────────────────────────────
        gamma = float(getattr(args, "gamma", 0.5))
        assert 0.0 <= gamma <= 1.0, "--gamma must be in [0, 1]"

        logging.info(
            f"[MUKSB] KS bargaining | gamma={gamma} "
            f"({'symmetric' if gamma == 0.5 else 'retain-favoured' if gamma > 0.5 else 'forget-favoured'}) "
            f"| method={args.method} | mask={'yes' if mask else 'no'} "
            f"| n_gpus={n_gpus}"
        )

        # Flat mask aligned with ``model.named_parameters()`` order ----------
        if mask is not None:
            mask_aligned = mask if use_dp else _strip_module(mask)
            mask_flat = build_flat_mask(
                model.named_parameters(), mask_aligned, self.device
            )
        else:
            mask_flat = None

        skipped_steps = 0
        total_steps = 0
        cos_phi_history = []

        model.train()
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            total_steps += 1

            # ── retain stage ──────────────────────────────────────────────────
            remain_x, remain_c = next(D_remain_iter)
            n_r = remain_x.size(0)
            remain_x = remain_x.to(self.device, non_blocking=True)
            remain_c = remain_c.to(self.device, non_blocking=True)
            remain_x = data_transform(self.config, remain_x)
            e_r = torch.randn_like(remain_x)
            b = self.betas

            t_r = torch.randint(
                low=0, high=self.num_timesteps, size=(n_r // 2 + 1,)
            ).to(self.device)
            t_r = torch.cat([t_r, self.num_timesteps - t_r - 1], dim=0)[:n_r]
            remain_loss = loss_registry_conditional[config.model.type](
                model, remain_x, t_r, remain_c, e_r, b
            )

            # ── forget stage ──────────────────────────────────────────────────
            forget_x, forget_c = next(D_forget_iter)
            n_f = forget_x.size(0)
            forget_x = forget_x.to(self.device, non_blocking=True)
            forget_c = forget_c.to(self.device, non_blocking=True)
            forget_x = data_transform(self.config, forget_x)
            e_f = torch.randn_like(forget_x)

            t_f = torch.randint(
                low=0, high=self.num_timesteps, size=(n_f // 2 + 1,)
            ).to(self.device)
            t_f = torch.cat([t_f, self.num_timesteps - t_f - 1], dim=0)[:n_f]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t_f, forget_c, e_f, b
                )
            else:
                # ``rl`` (default): match the noise prediction under a pseudo class
                a_f = (1 - b).cumprod(dim=0).index_select(0, t_f).view(-1, 1, 1, 1)
                forget_x_noisy = forget_x * a_f.sqrt() + e_f * (1.0 - a_f).sqrt()
                output = model(forget_x_noisy, t_f.float(), forget_c, mode="train")

                pseudo_c = torch.full(
                    forget_c.shape,
                    (args.label_to_forget + 1) % config.data.n_classes,
                    device=forget_c.device,
                )
                pseudo = model(
                    forget_x_noisy, t_f.float(), pseudo_c, mode="train"
                ).detach()
                forget_loss = criteria(pseudo, output)

            # ── compute gradients (no graph reuse — independent backwards) ────
            grads_r = torch.autograd.grad(
                remain_loss, model.parameters(), retain_graph=False
            )
            gr_flat = flatten_grads(model.parameters(), grads_r)
            del grads_r

            grads_f = torch.autograd.grad(
                forget_loss, model.parameters(), retain_graph=False
            )
            gf_flat = flatten_grads(model.parameters(), grads_f)
            del grads_f

            # ── restrict KS bargaining to masked subspace ─────────────────────
            if mask_flat is not None:
                gr_input = gr_flat[mask_flat]
                gf_input = gf_flat[mask_flat]
            else:
                gr_input = gr_flat
                gf_input = gf_flat

            # ── KS bargaining merge ───────────────────────────────────────────
            lambda_ks, cos_phi, g_star, effective_scale = ks_step(
                gr_input, gf_input, gamma=gamma,
            )
            cos_phi_history.append(cos_phi.item())

            # Anti-parallel: skip update entirely
            if torch.norm(g_star).item() < 1e-6:
                skipped_steps += 1
                del gr_flat, gf_flat, gr_input, gf_input, g_star
                if (step + 1) % self.config.training.log_freq == 0:
                    logging.info(
                        f"step: {step}, anti-parallel gradients "
                        f"(cos_phi={cos_phi.item():.3f}), skipping update"
                    )
                continue

            g_star_scaled = effective_scale * g_star
            del gr_input, gf_input

            # ── expand back to full parameter space ──────────────────────────
            if mask_flat is not None:
                update_full = torch.zeros_like(gr_flat)
                update_full[mask_flat] = g_star_scaled
            else:
                update_full = g_star_scaled
            del gr_flat, gf_flat, g_star, g_star_scaled

            # ── write merged update into model gradients & step ──────────────
            optimizer.zero_grad()
            unpack_to_grads(model.parameters(), update_full)
            del update_full

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                skip_rate = skipped_steps / max(total_steps, 1)
                avg_cos = float(np.mean(cos_phi_history[-self.config.training.log_freq:]))
                logging.info(
                    f"step: {step}, "
                    f"loss_r: {remain_loss.item():.4f}, "
                    f"loss_u: {forget_loss.item():.4f}, "
                    f"lambda_KS: {lambda_ks.item():.4f}, "
                    f"cos_phi: {cos_phi.item():.4f}, "
                    f"avg_cos_phi: {avg_cos:.4f}, "
                    f"eff_scale: {effective_scale.item():.4e}, "
                    f"skip_rate: {skip_rate:.3f}, "
                    f"time: {end - start:.2f}"
                )
                start = time.time()

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states_out = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states_out.append(ema_helper.state_dict())

                torch.save(
                    states_out,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

        # ── final diagnostics ─────────────────────────────────────────────────
        final_skip_rate = skipped_steps / max(total_steps, 1)
        logging.info(
            f"[MUKSB] Anti-parallel skips: {skipped_steps}/{total_steps} "
            f"({final_skip_rate * 100:.1f}%) — "
            f"high skip_rate indicates severe gradient conflict; consider increasing gamma"
        )
        if cos_phi_history:
            logging.info(
                f"[MUKSB] cos_phi stats: mean={np.mean(cos_phi_history):.4f}  "
                f"min={np.min(cos_phi_history):.4f}  "
                f"max={np.max(cos_phi_history):.4f}"
            )

    def load_ema_model(self):
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        return model

    def sample(self):
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )

        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        test_model = locals().get("test_model", model)

        if self.args.mode == "sample_fid":
            self.sample_fid(test_model, self.args.cond_scale)
        elif self.args.mode == "sample_classes":
            self.sample_classes(test_model, self.args.cond_scale)
        elif self.args.mode == "visualization":
            self.sample_visualization(
                model, str(self.args.cond_scale), self.args.cond_scale
            )

    def sample_classes(self, model, cond_scale):
        """
        Samples each class from the model. Can be used to calculate FIM, for generative replay
        or for classifier evaluation.

        Output layout (when forget_label >= 0):
          <ckpt_folder>/class{forget_label}_forget/
            class{gen_label}_images/   <- individual .png files
            canvas_class{gen_label}.png  <- grid of all generated images

        When forget_label == -1 (unknown), falls back to:
          <ckpt_folder>/class_samples/<gen_label>/
        """
        config = self.config
        args = self.args
        forget_label = getattr(args, "forget_label", -1)

        if forget_label >= 0:
            top_dir = os.path.join(args.ckpt_folder, f"class{forget_label}_forget")
        else:
            top_dir = os.path.join(args.ckpt_folder, "class_samples")
        os.makedirs(top_dir, exist_ok=True)

        classes, _ = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        for i in classes:
            if forget_label >= 0:
                img_dir = os.path.join(top_dir, f"class{i}_images")
            else:
                img_dir = os.path.join(top_dir, str(i))
            os.makedirs(img_dir, exist_ok=True)

            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class
            img_id = 0
            all_imgs = []

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating images — forget={forget_label}, gen_class={i}",
                ):
                    n = min(n_left, config.sampling.batch_size)
                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(img_dir, f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    all_imgs.append(x.cpu())
                    n_left -= n

            # build and save canvas
            all_imgs = torch.cat(all_imgs, dim=0)
            n_canvas = min(len(all_imgs), 100)  # cap at 100 images for canvas
            canvas_imgs = all_imgs[:n_canvas]
            nrow = min(10, n_canvas)
            grid = tvu.make_grid(canvas_imgs, nrow=nrow, normalize=True, padding=2)
            if forget_label >= 0:
                canvas_path = os.path.join(top_dir, f"canvas_class{i}.png")
            else:
                canvas_path = os.path.join(top_dir, f"canvas_{i}.png")
            tvu.save_image(grid, canvas_path)
            logging.info(f"Canvas saved: {canvas_path}")

    def sample_one_class(self, model, cond_scale, class_label):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_" + str(class_label))
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        total_n_samples = 500

        if total_n_samples % config.sampling.batch_size == 0:
            n_rounds = total_n_samples // config.sampling.batch_size
        else:
            n_rounds = total_n_samples // config.sampling.batch_size + 1
        n_left = total_n_samples  # tracker on how many samples left to generate

        with torch.no_grad():
            for j in tqdm.tqdm(
                range(n_rounds),
                desc=f"Generating image samples for class {class_label}",
            ):
                if n_left >= config.sampling.batch_size:
                    n = config.sampling.batch_size
                else:
                    n = n_left

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                c = torch.ones(x.size(0), device=self.device, dtype=int) * class_label
                x = self.sample_image(x, model, c, cond_scale)
                x = inverse_data_transform(config, x)

                for k in range(n):
                    tvu.save_image(
                        x[k], os.path.join(sample_dir, f"{img_id}.png"), normalize=True
                    )
                    img_id += 1

                n_left -= n

    def sample_fid(self, model, cond_scale):
        config = self.config
        args = self.args
        img_id = 0

        classes, excluded_classes = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        sample_dir = f"fid_samples_guidance_{args.cond_scale}"
        if excluded_classes:
            excluded_classes_str = "_".join(str(i) for i in excluded_classes)
            sample_dir = f"{sample_dir}_excluded_class_{excluded_classes_str}"
        sample_dir = os.path.join(args.ckpt_folder, sample_dir)
        os.makedirs(sample_dir, exist_ok=True)

        for i in classes:
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {i} for FID",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n

    def sample_image(self, x, model, c, cond_scale, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_conditional

            xs = generalized_steps_conditional(
                x, c, seq, model, self.betas, cond_scale, eta=self.args.eta
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_conditional

            x = ddpm_steps_conditional(x, c, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_visualization(self, model, name, cond_scale):
        config = self.config
        total_n_samples = config.training.visualization_samples
        assert total_n_samples % config.data.n_classes == 0
        n_rounds = (
            total_n_samples // config.sampling.batch_size
            if config.sampling.batch_size < total_n_samples
            else 1
        )

        # esd
        # c = torch.repeat_interleave(torch.arange(config.data.n_classes), total_n_samples//config.data.n_classes)
        c = torch.repeat_interleave(
            torch.arange(config.data.n_classes),
            total_n_samples // config.data.n_classes,
        ).to(self.device)

        c_chunks = torch.chunk(c, n_rounds, dim=0)

        with torch.no_grad():
            all_imgs = []
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for visualization."
            ):
                c = c_chunks[i]
                n = c.size(0)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, c, cond_scale)
                x = inverse_data_transform(config, x)

                all_imgs.append(x)

            all_imgs = torch.cat(all_imgs)
            grid = tvu.make_grid(
                all_imgs,
                nrow=total_n_samples // config.data.n_classes,
                normalize=True,
                padding=0,
            )

            try:
                tvu.save_image(
                    grid, os.path.join(self.config.log_dir, f"sample-{name}.png")
                )  # if called during training of base model
            except AttributeError:
                tvu.save_image(
                    grid, os.path.join(self.args.ckpt_folder, f"sample-{name}.png")
                )  # if called from sample.py

    def generate_mask(self):
        args, config = self.args, self.config
        logging.info(
            f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )

        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)
        
        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        optimizer = get_optimizer(config, model.parameters())

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = 0

        model.eval()

        for x, forget_c in D_forget_loader:
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            output = model(
                x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
            )

            # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/models/diffusion.py#L338
            loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient = param.grad.data.cpu()
                        gradients[name] += gradient

        with torch.no_grad():

            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

            mask_path = os.path.join('results/cifar10/mask', str(args.label_to_forget))
            os.makedirs(mask_path, exist_ok=True)

            threshold_list = [0.5]
            for i in threshold_list:
                print(i)
                sorted_dict_positions = {}
                hard_dict = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat(
                    [tensor.flatten() for tensor in gradients.values()]
                )

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index : start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements

                torch.save(hard_dict, os.path.join(mask_path, f'with_{str(i)}.pt'))
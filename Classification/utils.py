"""
Classification/utils.py — MUKSB
Model + dataset setup utilities.
Delegates to MUNBa Classification for model definitions, imagenet helpers, etc.
"""
import copy
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torchvision import transforms

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset import (
    TinyImageNet, cifar10_dataloaders, cifar100_dataloaders,
    svhn_dataloaders, celeba_dataloaders,
    cifar10_dataloaders_no_val, cifar100_dataloaders_no_val,
)
from models import model_dict
from models.resnet_im import resnet34

try:
    from imagenet import prepare_data
except ImportError:
    prepare_data = None


__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
    "dataset_convert_to_test",
    "dataset_convert_to_train",
]


# ─────────────────────────────────────────────────────────────────────────────
# Running average meter
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation module
# ─────────────────────────────────────────────────────────────────────────────

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean)
        if not isinstance(std,  torch.Tensor): std  = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, tensor):
        mean = self.mean[None, :, None, None]
        std  = self.std[None,  :, None, None]
        return tensor.sub(mean).div(std)

    def extra_repr(self): return f"mean={self.mean}, std={self.std}"


# ─────────────────────────────────────────────────────────────────────────────
# Seed / transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_seed(seed):
    print(f"Setup random seed = {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step
    lr = min(args.lr * current_steps / overall_steps, args.lr)
    for p in optimizer.param_groups:
        p["lr"] = lr


def dataset_convert_to_train(dataset, args=None):
    if args.dataset in ("cifar10", "svhn", "cifar100"):
        t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = t
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        t = transforms.Compose([])
    elif args.dataset == "celeba":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize])
    else:
        t = transforms.Compose([transforms.ToTensor()])
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = t
    dataset.train = False


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar"))


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print(f"Load checkpoint from: {filepath}")
        return torch.load(filepath, map_location=device, weights_only=False)
    print(f"Checkpoint not found: {filepath}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk]


# ─────────────────────────────────────────────────────────────────────────────
# Model + dataset setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_model_dataset(args):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed, only_mark=True, shuffle=True, no_aug=args.no_aug)
        setup_seed(args.train_seed or args.seed)
        model = model_dict[args.arch](num_classes=classes, imagenet=True if args.imagenet_arch else False)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "svhn":
        classes = 10
        normalization = NormalizeByChannelMeanStd([0.4377, 0.4438, 0.4728], [0.1201, 0.1231, 0.1052])
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed, only_mark=True, shuffle=True)
        model = model_dict[args.arch](num_classes=classes, imagenet=True if args.imagenet_arch else False)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed, only_mark=True, shuffle=True, no_aug=args.no_aug)
        model = model_dict[args.arch](num_classes=classes, imagenet=True if args.imagenet_arch else False)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_full_loader, val_loader, test_loader = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, _ = TinyImageNet(args).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed, only_mark=True, shuffle=True)
        model = model_dict[args.arch](num_classes=classes, imagenet=True if args.imagenet_arch else False)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, marked_loader

    elif args.dataset == "celeba":
        classes = 307
        normalization = NormalizeByChannelMeanStd([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_full_loader, val_loader, test_loader, forget_loader, retain_loader = celeba_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed, only_mark=True, shuffle=True,
            forget_fraction=getattr(args, "forget_fraction", 0.1))
        setup_seed(args.train_seed or args.seed)
        imagenet_weights = getattr(args, "imagenet_weights", None)
        if imagenet_weights:
            # Initialise from a converted torchvision-format ImageNet backbone
            # (e.g. microsoft/resnet-34 via convert_hf_resnet34.py) instead of
            # the default torchvision ImageNet weights.
            model = resnet34(pretrained=False)
            ckpt = torch.load(imagenet_weights, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            # Only the classifier head may be missing/unexpected (it is replaced below).
            assert all("fc" in k for k in list(missing) + list(unexpected)), (
                f"Unexpected backbone key mismatch loading {imagenet_weights}: "
                f"missing={missing}, unexpected={unexpected}"
            )
            print(f"Initialised CelebA ResNet-34 from ImageNet backbone: {imagenet_weights}")
        else:
            model = resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
        model.normalize = normalization
        return model, train_full_loader, val_loader, test_loader, forget_loader, retain_loader

    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported!")


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)


def get_unlearn_loader(marked_loader, args):
    forget_dataset  = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data    = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(forget_dataset, args.batch_size, args.seed, True)
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data    = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(retain_dataset, args.batch_size, args.seed, True)
    print(f"Forget: {len(forget_dataset)}, Retain: {len(retain_dataset)}")
    return forget_loader, retain_loader


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if not commands: return
    if os.path.exists(dir): shutil.rmtree(dir)
    if shuffle: random.shuffle(commands); random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)
    with open(f"stop_{dir}.sh", "w") as f:
        print(f"kill $(ps aux|grep 'bash {dir}'|awk '{{print $2}}')", file=f)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::len(gpus)]
        if not i_commands: continue
        sh_path = os.path.join(dir, f"run{i}.sh")
        with open(sh_path, "w") as f:
            for com in i_commands:
                print(f"CUDA_VISIBLE_DEVICES={gpu} {com}", file=f)
        if call:
            os.system(f"bash {sh_path}&")
            time.sleep(delay)

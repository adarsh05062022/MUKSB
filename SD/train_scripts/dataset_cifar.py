import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

############################################
# CIFAR10 CLASS NAMES
############################################

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


############################################
# IMAGE TRANSFORM
############################################
def add_texture_noise(x, strength=0.03):
    noise = torch.randn_like(x)
    high_freq = noise - torch.nn.functional.avg_pool2d(noise, 3, stride=1, padding=1)
    return x + strength * high_freq


def get_transform(image_size=512):

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomAdjustSharpness(2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_texture_noise(x)),
        transforms.Normalize([0.5],[0.5])
    ])


############################################
# MODEL SETUP
############################################

def setup_model(config, ckpt, device):

    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd

    model = instantiate_from_config(config.model)

    model.load_state_dict(sd, strict=False)

    model.to(device)

    model.eval()

    model.cond_stage_model.device = device

    return model


############################################
# REMAIN DATA
############################################

def setup_remain_data(
        class_to_forget,
        batch_size,
        image_size,
        root="./datasets"
):

    transform = get_transform(image_size)

    dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    remain_indices = [
        i for i, label in enumerate(dataset.targets)
        if label != class_to_forget
    ]

    remain_dataset = Subset(dataset, remain_indices)

    descriptions = [
        f"an image of a {cls}" for cls in CIFAR10_CLASSES
    ]

    loader = DataLoader(
        remain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader, descriptions


############################################
# FORGET DATA
############################################

def setup_forget_data(
        class_to_forget,
        batch_size,
        image_size,
        root="./datasets"
):

    transform = get_transform(image_size)

    dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    forget_indices = [
        i for i, label in enumerate(dataset.targets)
        if label == class_to_forget
    ]

    forget_dataset = Subset(dataset, forget_indices)

    descriptions = [
        f"an image of a {cls}" for cls in CIFAR10_CLASSES
    ]

    loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader, descriptions
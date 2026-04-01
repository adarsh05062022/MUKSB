#datset_food.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

############################################
# FOOD-101 → 10 CLASS SUBSET
############################################

FOOD10_CLASSES = [
    "pizza",
    "sushi",
    "hamburger",
    "hot_and_sour_soup",
    "ice_cream",
    "chocolate_cake",
    "ramen",
    "steak",
    "tacos",
    "waffles",
]

# Maps Food101's string class names → our 0-9 integer labels
FOOD10_CLASS_TO_IDX = {name: i for i, name in enumerate(FOOD10_CLASSES)}


############################################
# IMAGE TRANSFORM — clean, no noise/sharpening
############################################

def get_transform(image_size=512):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, remapped_labels):
        self.dataset = dataset
        self.indices = indices
        self.remapped_labels = remapped_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.dataset[real_idx]
        label = self.remapped_labels[real_idx]
        return img, label
    
############################################
# INTERNAL: build filtered dataset + label map
############################################

def _build_food10_dataset(root, image_size):
    """
    Returns (dataset, remapped_labels) where:
    - dataset: full Food101 torchvision dataset
    - remapped_labels: list of int (0-9) or -1 (not in FOOD10)
    """
    transform = get_transform(image_size)
    dataset = Food101(
        root=root,
        split="train",
        download=False,
        transform=transform
    )

    # dataset.classes: list of 101 class name strings from Food101
    # dataset._labels: list of int indices into dataset.classes
    remapped = []
    for raw_label in dataset._labels:
        class_name = dataset.classes[raw_label]        # e.g. "pizza"
        remapped.append(FOOD10_CLASS_TO_IDX.get(class_name, -1))

    return dataset, remapped


############################################
# MODEL SETUP  (unchanged from CIFAR version)
############################################

def setup_model(config, ckpt, device):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

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
        class_to_forget,   # int 0-9 into FOOD10_CLASSES
        batch_size,
        image_size,
        root="./datasets"
):
    dataset, remapped_labels = _build_food10_dataset(root, image_size)

    remain_indices = [
        i for i, label in enumerate(remapped_labels)
        if label != -1 and label != class_to_forget      # only our 10 classes, excluding forget class
    ]

    remain_dataset = RemappedSubset(dataset, remain_indices, remapped_labels)

    descriptions = [f"an image of {cls}" for cls in FOOD10_CLASSES]

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
        class_to_forget,   # int 0-9 into FOOD10_CLASSES
        batch_size,
        image_size,
        root="./datasets"
):
    dataset, remapped_labels = _build_food10_dataset(root, image_size)

    forget_indices = [
        i for i, label in enumerate(remapped_labels)
        if label == class_to_forget
    ]

    forget_dataset = RemappedSubset(dataset, forget_indices, remapped_labels)

    descriptions = [f"an image of {cls}" for cls in FOOD10_CLASSES]

    loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader, descriptions
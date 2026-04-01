"""
CLIP/loadData/dataset.py — MUKSB
DataLoader factories for Oxford Pets and other vision-language datasets.
10% of classes are chosen as the forget set; the rest are retain.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from typing import Sequence, TypeVar

from .load_oxfordpets import OxfordPets

T_co = TypeVar("T_co", covariant=True)


class Custom_Subset(Dataset[T_co]):
    """Subset that also exposes a .targets attribute."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = np.array(dataset.targets)[indices]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]


class preprocessDataset(torch.utils.data.Dataset):
    """Wrap a dataset with a custom transform applied at fetch time."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return self.transform(image), target


def oxfordPets_dataloaders(batch_size=128, data_dir="/data/oxfordpets",
                           num_workers=2, seed=1, no_aug=False):
    """
    Oxford Pets dataloaders with 10% class forget / 90% class retain split.

    Returns
    -------
    train_loader, val_loader, test_loader, forget_loader, retain_loader, class_name
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = (transforms.Compose([
                           transforms.Resize((256, 256)), transforms.CenterCrop(224),
                           transforms.ToTensor(), normalize])
                       if no_aug else
                       transforms.Compose([
                           transforms.Resize((256, 256)), transforms.CenterCrop(224),
                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])

    print("Dataset: Oxford Pets")
    train_set = OxfordPets(data_dir, train=True,  transform=train_transform)
    test_set  = OxfordPets(data_dir, train=False, transform=test_transform)
    train_set.targets = np.array(train_set.targets)
    test_set.targets  = np.array(test_set.targets)
    class_name = train_set.unique_breeds

    # 10% of classes → forget; rest → retain
    unl_targets = np.random.choice(np.unique(train_set.targets),
                                   int(0.1 * len(np.unique(train_set.targets))), replace=False)
    rem_targets  = np.setdiff1d(np.unique(train_set.targets), unl_targets)
    print(f"Forget classes: {unl_targets}  |  Retain classes: {len(rem_targets)}")

    unl_idx = np.where(np.isin(train_set.targets, unl_targets))[0]
    rem_idx = np.where(np.isin(train_set.targets, rem_targets))[0]
    forget_set      = Custom_Subset(train_set, unl_idx)
    remain_set      = Custom_Subset(train_set, rem_idx)
    test_remain_idx = np.where(np.isin(test_set.targets, rem_targets))[0]
    test_remain_set = Custom_Subset(test_set,  test_remain_idx)

    def _init_fn(w): np.random.seed(int(seed))
    kw = dict(num_workers=0, pin_memory=False,
              worker_init_fn=_init_fn if seed is not None else None)
    print(f"Train: {len(train_set)}, Test: {len(test_set)}, Classes: {len(class_name)}")
    return (DataLoader(train_set,       batch_size, shuffle=True,  **kw),
            DataLoader(remain_set,      batch_size, shuffle=False, **kw),
            DataLoader(test_remain_set, batch_size, shuffle=False, **kw),
            DataLoader(forget_set,      batch_size, shuffle=True,  **kw),
            DataLoader(remain_set,      batch_size, shuffle=True,  **kw),
            class_name)

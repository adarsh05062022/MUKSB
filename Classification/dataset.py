"""
Classification/dataset.py — MUKSB
Dataset loaders for CIFAR-10, CIFAR-100, SVHN, TinyImageNet, CelebA.
Exact copy of MUNBa/Classification/dataset.py; shared via this project.
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from tqdm import tqdm

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from load_celeba import CelebA


def cifar10_dataloaders_no_val(batch_size=128, data_dir="datasets/cifar10", num_workers=2):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform  = transforms.Compose([transforms.ToTensor()])
    print("Dataset: CIFAR-10 (no val split)")
    train_set = CIFAR10(data_dir, train=True,  transform=train_transform, download=True)
    val_set   = CIFAR10(data_dir, train=False, transform=test_transform,  download=True)
    test_set  = CIFAR10(data_dir, train=False, transform=test_transform,  download=True)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
            DataLoader(val_set,   batch_size, shuffle=False, **kw),
            DataLoader(test_set,  batch_size, shuffle=False, **kw))


def cifar100_dataloaders_no_val(batch_size=128, data_dir="datasets/cifar100", num_workers=2):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform  = transforms.Compose([transforms.ToTensor()])
    print("Dataset: CIFAR-100 (no val split)")
    train_set = CIFAR100(data_dir, train=True,  transform=train_transform, download=True)
    val_set   = CIFAR100(data_dir, train=False, transform=test_transform,  download=True)
    test_set  = CIFAR100(data_dir, train=False, transform=test_transform,  download=True)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
            DataLoader(val_set,   batch_size, shuffle=False, **kw),
            DataLoader(test_set,  batch_size, shuffle=False, **kw))


def svhn_dataloaders(batch_size=128, data_dir="datasets/svhn", num_workers=2,
                     class_to_replace=None, num_indexes_to_replace=None,
                     indexes_to_replace=None, seed=1, only_mark=False, shuffle=True, no_aug=False):
    transform = transforms.Compose([transforms.ToTensor()])
    print("Dataset: SVHN")
    train_set = SVHN(data_dir, split="train", transform=transform, download=True)
    test_set  = SVHN(data_dir, split="test",  transform=transform, download=True)
    train_set.labels = np.array(train_set.labels)
    test_set.labels  = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)
    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]
    train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_set.data   = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of class_to_replace and indexes_to_replace can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace, seed - 1, only_mark)
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data   = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(train_set, indexes_to_replace, seed - 1, only_mark)

    def _init_fn(worker_id): np.random.seed(int(seed))
    kw = dict(num_workers=0, pin_memory=False, worker_init_fn=_init_fn if seed is not None else None)
    return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
            DataLoader(valid_set, batch_size, shuffle=False, **kw),
            DataLoader(test_set,  batch_size, shuffle=False, **kw))


def cifar100_dataloaders(batch_size=128, data_dir="/datasets/CIFAR100", num_workers=2,
                         class_to_replace=None, num_indexes_to_replace=None,
                         indexes_to_replace=None, seed=1, only_mark=False, shuffle=True, no_aug=False):
    train_transform = (transforms.Compose([transforms.ToTensor()]) if no_aug
                       else transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]))
    test_transform = transforms.Compose([transforms.ToTensor()])
    print("Dataset: CIFAR-100")
    train_set = CIFAR100(data_dir, train=True,  transform=train_transform, download=True)
    test_set  = CIFAR100(data_dir, train=False, transform=test_transform,  download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets  = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)
    valid_set.data    = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]
    train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_set.data    = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of class_to_replace and indexes_to_replace can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace, seed - 1, only_mark)
        if num_indexes_to_replace is None:
            test_set.data    = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(train_set, indexes_to_replace, seed - 1, only_mark)

    def _init_fn(worker_id): np.random.seed(int(seed))
    kw = dict(num_workers=0, pin_memory=False, worker_init_fn=_init_fn if seed is not None else None)
    return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
            DataLoader(valid_set, batch_size, shuffle=False, **kw),
            DataLoader(test_set,  batch_size, shuffle=False, **kw))


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):  return len(self.targets)
    def __getitem__(self, idx):
        return (self.transform(self.imgs[idx]) if self.transform else self.imgs[idx]), self.targets[idx]


class TinyImageNet:
    def __init__(self, args, normalize=False):
        self.args = args
        self.norm_layer = (transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           if normalize else None)
        self.tr_train = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                            transforms.RandomHorizontalFlip()])
        self.tr_test  = transforms.Compose([])
        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path   = os.path.join(args.data_dir, "val/")
        self.test_path  = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            with open(os.path.join(self.val_path, "val_annotations.txt")) as f:
                for line in f:
                    s = line.split("\t"); val_dict[s[0]] = s[1]
            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]; folder = val_dict[file]
                for d in [self.val_path + str(folder), self.test_path + str(folder)]:
                    if not os.path.exists(d):
                        os.mkdir(d); os.mkdir(d + "/images")
            for path in paths:
                file = path.split("/")[-1]; folder = val_dict[file]
                dest = ((self.val_path if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25
                         else self.test_path) + str(folder) + "/images/" + file)
                move(path, dest)
            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(self, batch_size=128, data_dir="datasets/tiny", num_workers=2,
                     class_to_replace=None, num_indexes_to_replace=None,
                     indexes_to_replace=None, seed=1, only_mark=False, shuffle=True, no_aug=False):
        train_set = TinyImageNetDataset(ImageFolder(self.train_path, transform=self.tr_train), self.norm_layer)
        test_set  = TinyImageNetDataset(ImageFolder(self.test_path,  transform=self.tr_test),  self.norm_layer)
        train_set.targets = np.array(train_set.targets)

        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False))
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)
        valid_set.imgs    = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]
        train_idx = list(set(range(len(train_set))) - set(valid_idx))
        train_set.imgs    = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError("Only one of class_to_replace / indexes_to_replace can be specified")
        if class_to_replace is not None:
            replace_class(train_set, class_to_replace, num_indexes_to_replace, seed - 1, only_mark)
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs    = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(train_set, indexes_to_replace, seed - 1, only_mark)

        def _init_fn(worker_id): np.random.seed(int(seed))
        kw = dict(num_workers=0, pin_memory=False, worker_init_fn=_init_fn if seed is not None else None)
        print(f"Train: {len(train_set)}, Test: {len(test_set)}")
        return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
                DataLoader(test_set,  batch_size, shuffle=False, **kw),
                DataLoader(test_set,  batch_size, shuffle=False, **kw))


def cifar10_dataloaders(batch_size=128, data_dir="/datasets/CIFAR10", num_workers=2,
                        random_to_replace=None, class_to_replace=None,
                        num_indexes_to_replace=None, indexes_to_replace=None,
                        seed=1, only_mark=False, shuffle=True, no_aug=False):
    train_transform = (transforms.Compose([transforms.ToTensor()]) if no_aug
                       else transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]))
    test_transform = transforms.Compose([transforms.ToTensor()])
    print("Dataset: CIFAR-10  |  45k train, 5k val, 10k test")

    train_set = CIFAR10(data_dir, train=True,  transform=train_transform, download=True)
    test_set  = CIFAR10(data_dir, train=False, transform=test_transform,  download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets  = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)
    valid_set.data    = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]
    train_idx = list(set(range(len(train_set))) - set(valid_idx))
    train_set.data    = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of class_to_replace / indexes_to_replace can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace, seed - 1, only_mark)
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data    = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(train_set, indexes_to_replace, seed - 1, only_mark)

    def _init_fn(worker_id): np.random.seed(int(seed))
    kw = dict(num_workers=0, pin_memory=False, worker_init_fn=_init_fn if seed is not None else None)
    return (DataLoader(train_set, batch_size, shuffle=True,  **kw),
            DataLoader(valid_set, batch_size, shuffle=False, **kw),
            DataLoader(test_set,  batch_size, shuffle=False, **kw))


# ── Forget/retain helpers ──────────────────────────────────────────────────────

def replace_indexes(dataset, indexes, seed=0, only_mark=False):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        try:    dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            try: dataset.labels[indexes]  = dataset.labels[new_indexes]
            except: dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        try:    dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try: dataset.labels[indexes]  = -dataset.labels[indexes]  - 1
            except: dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(dataset, class_to_replace, num_indexes_to_replace=None, seed=0, only_mark=False):
    if class_to_replace == -1:
        try:    indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try: indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except: indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:    indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try: indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except: indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(indexes)
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
    replace_indexes(dataset, indexes, seed, only_mark)


def celeba_dataloaders(batch_size=128, data_dir="/home/jing/dataset/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset",
                       num_workers=2, random_to_replace=None, class_to_replace=None,
                       num_indexes_to_replace=None, indexes_to_replace=None,
                       seed=1, only_mark=False, shuffle=True, no_aug=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = (transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                           transforms.ToTensor(), normalize]) if no_aug else
                       transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                         transforms.ToTensor(), normalize])
    print("Dataset: CelebA Mask HQ")
    train_set = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    test_set  = ImageFolder(os.path.join(data_dir, "test"),  transform=test_transform)
    train_set.targets = np.array(train_set.targets)
    test_set.targets  = np.array(test_set.targets)

    unl_ids = np.random.choice(np.unique(train_set.targets),
                                int(0.1 * len(np.unique(train_set.targets))), replace=False)
    rem_ids  = np.setdiff1d(np.unique(train_set.targets), unl_ids)
    forget_set  = Subset(train_set, np.where(np.isin(train_set.targets, unl_ids))[0])
    remain_set  = Subset(train_set, np.where(np.isin(train_set.targets, rem_ids))[0])
    test_remain = Subset(test_set,  np.where(np.isin(test_set.targets,  rem_ids))[0])

    def _init_fn(w): np.random.seed(int(seed))
    kw = dict(num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn if seed is not None else None)
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    return (DataLoader(train_set,   batch_size, shuffle=True,  **kw),
            DataLoader(remain_set,  batch_size, shuffle=False, **kw),
            DataLoader(test_remain, batch_size, shuffle=False, **kw),
            DataLoader(forget_set,  batch_size, shuffle=True,  **kw),
            DataLoader(remain_set,  batch_size, shuffle=True,  **kw))

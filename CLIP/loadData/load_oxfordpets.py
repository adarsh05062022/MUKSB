"""
CLIP/loadData/load_oxfordpets.py — MUKSB
Oxford-IIIT Pets dataset class (37 breeds, 80/20 train/test split).
"""
import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class OxfordPets(Dataset):
    """
    Oxford-IIIT Pets dataset.
    Loads images from <root>/images/*.jpg and infers breed from filename
    (everything before the last underscore+number).
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        img_path   = os.path.join(root, "images", "*.jpg")
        pets_files  = glob.glob(img_path)
        breed_names = ["_".join(f.split("/")[-1].split("_")[:-1]) for f in pets_files]
        unique_breeds = sorted(set(breed_names))
        print(f"OxfordPets: {len(unique_breeds)} breeds, {len(pets_files)} images  (root={root})")

        self.breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
        pets_targets = [self.breed_to_idx["_".join(f.split("/")[-1].split("_")[:-1])]
                        for f in pets_files]

        train_files, test_files, train_targets, test_targets = train_test_split(
            pets_files, pets_targets, test_size=0.2, random_state=42, stratify=pets_targets)

        img_files = train_files if train else test_files
        targets   = train_targets if train else test_targets

        self.data    = [Image.open(f).convert("RGB") for f in img_files]
        self.targets = targets
        self.unique_breeds = unique_breeds

    def __len__(self):   return len(self.data)

    def __getitem__(self, idx):
        img    = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

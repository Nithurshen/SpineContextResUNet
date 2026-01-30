import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class SpineDataset(Dataset):
    def __init__(self, split="train", root_dir="data/processed", transform=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, split, "images", "*.npy"))
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(root_dir, split, "masks", "*.npy"))
        )

        assert len(self.image_paths) == len(self.mask_paths), (
            "Mismatch between images and masks!"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if mask.ndim == 3:
            mask = mask[np.newaxis, ...]

        if self.transform:
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        mask = (mask > 0.5).float()

        return image, mask

    def _augment(self, image, mask):
        # 1. Random Flip (Left-Right)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        # 2. Random Flip (Up-Down)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 3. Random 90 degree rotation (XY plane)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        # 4. Intensity Shift (Simulates different scanner calibrations)
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            image = image + shift
            image = np.clip(image, 0, 1)

        return image, mask

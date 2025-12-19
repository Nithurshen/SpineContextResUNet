import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
PATCH_SIZE = (128, 128, 64)
NUM_PATCHES_PER_VOLUME = 20


def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata(), img.header.get_zooms()


def normalize_intensity(vol):
    min_hu, max_hu = -1000, 2000
    vol = np.clip(vol, min_hu, max_hu)
    vol = (vol - min_hu) / (max_hu - min_hu)
    return vol


def extract_patches(vol, mask, num_patches):
    vol_shape = vol.shape
    patches_img = []
    patches_mask = []

    fg_indices = np.argwhere(mask > 0)
    bg_indices = np.argwhere(mask == 0)

    if len(fg_indices) == 0:
        return [], []

    for i in range(num_patches):
        if np.random.rand() < 0.7 or len(bg_indices) == 0:
            center = fg_indices[np.random.choice(len(fg_indices))]
        else:
            center = bg_indices[np.random.choice(len(bg_indices))]

        d, h, w = PATCH_SIZE
        z_c, y_c, x_c = center

        z_start = max(0, min(z_c - d // 2, vol_shape[0] - d))
        y_start = max(0, min(y_c - h // 2, vol_shape[1] - h))
        x_start = max(0, min(x_c - w // 2, vol_shape[2] - w))

        img_patch = vol[
            z_start : z_start + d, y_start : y_start + h, x_start : x_start + w
        ]
        mask_patch = mask[
            z_start : z_start + d, y_start : y_start + h, x_start : x_start + w
        ]

        if img_patch.shape != PATCH_SIZE:
            pad_width = [
                (0, max(0, s - c)) for s, c in zip(PATCH_SIZE, img_patch.shape)
            ]
            img_patch = np.pad(img_patch, pad_width, mode="constant")
            mask_patch = np.pad(mask_patch, pad_width, mode="constant")

        patches_img.append(img_patch)
        patches_mask.append(mask_patch)

    return patches_img, patches_mask


def process_dataset():
    ct_files = sorted(
        glob.glob(os.path.join(RAW_DATA_DIR, "**/*ct.nii.gz"), recursive=True)
    )
    print(f"Found {len(ct_files)} CT volumes.")

    for ct_path in tqdm(ct_files):
        if "dataset-01training" in ct_path:
            split = "train"
        elif "dataset-02validation" in ct_path:
            split = "val"
        else:
            continue

        mask_path = ct_path.replace("rawdata", "derivatives").replace(
            "_ct.nii.gz", "_seg-vert_msk.nii.gz"
        )

        if not os.path.exists(mask_path):
            mask_path = ct_path.replace("rawdata", "derivatives").replace(
                "_ct.nii.gz", "_seg-vert_msk.nii"
            )

        if not os.path.exists(mask_path):
            mask_path = ct_path.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")

        if not os.path.exists(mask_path):
            print(f"Skipping {os.path.basename(ct_path)}: Mask NOT found.")
            continue

        try:
            vol, _ = load_nifti(ct_path)
            mask, _ = load_nifti(mask_path)

            vol = normalize_intensity(vol)
            img_patches, mask_patches = extract_patches(
                vol, mask, NUM_PATCHES_PER_VOLUME
            )

            base_name = os.path.basename(ct_path).split(".")[0]
            for i, (p_img, p_mask) in enumerate(zip(img_patches, mask_patches)):
                save_name = f"{base_name}_patch{i}.npy"
                np.save(os.path.join(PROCESSED_DIR, split, "images", save_name), p_img)
                np.save(os.path.join(PROCESSED_DIR, split, "masks", save_name), p_mask)
        except Exception as e:
            print(f"Error processing {ct_path}: {e}")


if __name__ == "__main__":
    process_dataset()

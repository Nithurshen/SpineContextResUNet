import os
import torch
import nibabel as nib
import numpy as np
import time
import glob
import tempfile
from totalsegmentator.python_api import totalsegmentator

# --- CONFIGURATION ---
DEVICE = "gpu" if torch.backends.mps.is_available() else "cpu"
VOL_PATH = "data/raw/dataset-03test/rawdata/sub-verse714/sub-verse714_dir-iso_ct.nii.gz"
MSK_PATH = "data/raw/dataset-03test/derivatives/sub-verse714/sub-verse714_dir-iso_seg-vert_msk.nii.gz"


def compute_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)


def run_test():
    print(f"Loading Ground Truth...")
    gt_nii = nib.load(MSK_PATH)
    gt_data = (gt_nii.get_fdata() > 0).astype(np.float32)

    devices_to_test = ["cpu"]
    if torch.backends.mps.is_available():
        devices_to_test.append("mps")

    for dev in devices_to_test:
        print(f"\n--- Testing TotalSegmentator on {dev.upper()} ---")

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                totalsegmentator(
                    VOL_PATH, tmp_dir, task="total", device=dev, quiet=True
                )

                vert_files = glob.glob(os.path.join(tmp_dir, "vertebrae_*.nii.gz"))
                if not vert_files:
                    print("No vertebrae detected.")
                    continue

                combined_mask = None
                for v_file in vert_files:
                    data = nib.load(v_file).get_fdata()
                    if combined_mask is None:
                        combined_mask = data
                    else:
                        combined_mask += data

                pred_bin = (combined_mask > 0).astype(np.float32)

            except Exception as e:
                print(f"Error during TS inference: {e}")
                continue

        elapsed = time.time() - start_time
        dice = compute_dice(pred_bin, gt_data)

        print(f"Results for TotalSegmentator ({dev.upper()}):")
        print(f"  Time Taken: {elapsed:.2f} seconds")
        print(f"  Dice Score: {dice:.4f}")


if __name__ == "__main__":
    run_test()

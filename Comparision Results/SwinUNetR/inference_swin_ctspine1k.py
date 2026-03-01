import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label

from SwinUNetR.model import SwinUNETR_Nano

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

MODEL_PATH = "models/best_swin_model.pth"
TEST_VOL_DIR = "data/raw/CTSpine1k/volumes/test"
TEST_SEG_DIR = "data/raw/CTSpine1k/labels/test"
RESULTS_DIR = "results/swin_inference/ctspine1k"

PATCH_SIZE = (128, 128, 64)
OVERLAP = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_gaussian_window(patch_size):
    d, h, w = patch_size
    z_win = torch.hann_window(d)
    y_win = torch.hann_window(h)
    x_win = torch.hann_window(w)

    z_y_win = torch.outer(z_win, y_win)
    window = torch.outer(z_y_win.flatten(), x_win).view(d, h, w)
    return window


def keep_largest_blob(mask):
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return mask
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    return (labeled_mask == largest_label).astype(np.float32)


def compute_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)


def compute_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def compute_recall(pred, gt):
    intersection = np.sum(pred * gt)
    return (intersection + 1e-6) / (np.sum(gt) + 1e-6)


def compute_precision(pred, gt):
    intersection = np.sum(pred * gt)
    return (intersection + 1e-6) / (np.sum(pred) + 1e-6)


def predict_sliding_window(model, vol):
    d, h, w = vol.shape
    pd, ph, pw = PATCH_SIZE

    prob_map = torch.zeros(vol.shape, device="cpu")
    weight_map = torch.zeros(vol.shape, device="cpu")
    patch_window = get_gaussian_window(PATCH_SIZE).to(DEVICE)

    stride_d, stride_h, stride_w = [int(p * (1 - OVERLAP)) for p in PATCH_SIZE]
    vol_t = torch.from_numpy(vol).float()

    z_steps = sorted(
        list(set(list(range(0, d - pd + stride_d, stride_d)) + [max(0, d - pd)]))
    )
    y_steps = sorted(
        list(set(list(range(0, h - ph + stride_h, stride_h)) + [max(0, h - ph)]))
    )
    x_steps = sorted(
        list(set(list(range(0, w - pw + stride_w, stride_w)) + [max(0, w - pw)]))
    )

    model.eval()

    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    slice_vol = vol_t[z : z + pd, y : y + ph, x : x + pw]
                    curr_d, curr_h, curr_w = slice_vol.shape

                    need_pad = False
                    if (curr_d, curr_h, curr_w) != PATCH_SIZE:
                        need_pad = True
                        pad_d = pd - curr_d
                        pad_h = ph - curr_h
                        pad_w = pw - curr_w
                        slice_vol = F.pad(
                            slice_vol.unsqueeze(0).unsqueeze(0),
                            (0, pad_w, 0, pad_h, 0, pad_d),
                        ).squeeze()

                    patch = slice_vol.unsqueeze(0).unsqueeze(0).to(DEVICE)

                    logits = model(patch)
                    pred_patch = torch.sigmoid(logits).squeeze()

                    weighted_pred = pred_patch * patch_window
                    weighted_win = patch_window.clone()

                    if need_pad:
                        weighted_pred = weighted_pred[:curr_d, :curr_h, :curr_w]
                        weighted_win = weighted_win[:curr_d, :curr_h, :curr_w]

                    prob_map[z : z + curr_d, y : y + curr_h, x : x + curr_w] += (
                        weighted_pred.cpu()
                    )
                    weight_map[z : z + curr_d, y : y + curr_h, x : x + curr_w] += (
                        weighted_win.cpu()
                    )

    weight_map[weight_map == 0] = 1.0
    avg_prob = prob_map / weight_map

    return avg_prob.numpy()


def save_visual(ct_vol, pred_mask, subject_id, output_dir):
    mid_idx = ct_vol.shape[0] // 2

    ct_slice = ct_vol[mid_idx, :, :].T
    mask_slice = pred_mask[mid_idx, :, :].T

    binary_mask = (mask_slice > 0.5).astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(ct_slice, cmap="gray", origin="lower")

    masked_pred = np.ma.masked_where(binary_mask == 0, binary_mask)
    ax.imshow(masked_pred, cmap="winter", alpha=0.5, origin="lower")

    ax.set_title(
        f"Swin Prediction: {subject_id}",
        fontsize=14,
        color="white",
        backgroundcolor="black",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{subject_id}_seg.png"), facecolor="black")
    plt.close()


def run_evaluation():
    print(f"--- Loading SwinUNETR-Nano on {DEVICE} ---")

    model = SwinUNETR_Nano(in_channels=1, out_channels=1).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    else:
        print(f"ERROR: Model weights not found at {MODEL_PATH}")
        return

    vol_files = sorted(glob.glob(os.path.join(TEST_VOL_DIR, "*.nii.gz")))

    if not vol_files:
        print(f"No volume files found in {TEST_VOL_DIR}")
        return

    detailed_results = []
    print(f"--- Processing {len(vol_files)} Volumes ---")
    print(
        f"{'Subject ID':<20} | {'Dice':<8} | {'IoU':<8} | {'Recall':<8} | {'Prec':<8}"
    )
    print("-" * 65)

    for vol_path in tqdm(vol_files, desc="Inference"):
        file_name = os.path.basename(vol_path)
        subject_id = file_name.replace(".nii.gz", "")

        potential_labels = glob.glob(
            os.path.join(TEST_SEG_DIR, f"{subject_id}*.nii.gz")
        )

        if not potential_labels:
            print(f"\nWarning: Label not found for {subject_id}. Skipping metrics.")
            continue

        label_path = potential_labels[0]

        vol_nii = nib.as_closest_canonical(nib.load(vol_path))
        gt_nii = nib.as_closest_canonical(nib.load(label_path))

        vol_data = np.clip(vol_nii.get_fdata(), -1000, 2000)
        vol_data = (vol_data + 1000) / 3000
        gt_data = (gt_nii.get_fdata() > 0).astype(np.float32)

        pred_prob = predict_sliding_window(model, vol_data)

        pred_bin = (pred_prob > 0.5).astype(np.float32)
        pred_bin = keep_largest_blob(pred_bin)

        dice = compute_dice(pred_bin, gt_data)
        iou = compute_iou(pred_bin, gt_data)
        recall = compute_recall(pred_bin, gt_data)
        precision = compute_precision(pred_bin, gt_data)

        detailed_results.append(
            {
                "ID": subject_id,
                "Dice": dice,
                "IoU": iou,
                "Recall": recall,
                "Precision": precision,
            }
        )

        save_visual(vol_data, pred_bin, subject_id, RESULTS_DIR)

        print(
            f"{subject_id:<20} | {dice:<8.4f} | {iou:<8.4f} | {recall:<8.4f} | {precision:<8.4f}"
        )

    if detailed_results:
        dices = [r["Dice"] for r in detailed_results]
        ious = [r["IoU"] for r in detailed_results]
        recalls = [r["Recall"] for r in detailed_results]
        precs = [r["Precision"] for r in detailed_results]

        print("\n" + "=" * 55)
        print("FINAL TEST SET PERFORMANCE SUMMARY (SwinUNETR)")
        print(f"Dice      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"IoU       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"Recall    : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"Best Dice : {np.max(dices):.4f}")
        print("=" * 55)

        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(RESULTS_DIR, "test_metrics_swin_ctspine1k.csv")
        df.to_csv(csv_path, index=False)
        print(f"Detailed metrics saved to: {csv_path}")
    else:
        print("No paired data found for evaluation.")


if __name__ == "__main__":
    run_evaluation()

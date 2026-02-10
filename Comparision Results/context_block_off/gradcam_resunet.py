import os

# --- CRITICAL FIX FOR MACOS / MPS ---
# Enables CPU fallback for operations not yet implemented on MPS (like MaxPool3d)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import gc
import matplotlib.pyplot as plt
from scipy.ndimage import label

# --- IMPORT MODEL FROM YOUR FILE ---
# Ensure model.py contains the SpineResUNet_cotext_off class
from model import SpineResUNet_cotext_off

# --- Configuration ---
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# UPDATED: Path for the Context-Off model
MODEL_PATH = "models/best_model_context_off.pth"

# Single Instance Paths
VOL_PATH = "data/raw/dataset-03test/rawdata/sub-verse803/sub-verse803_dir-iso_ct.nii.gz"
MSK_PATH = "data/raw/dataset-03test/derivatives/sub-verse803/sub-verse803_dir-iso_seg-vert_msk.nii.gz"

RESULTS_DIR = "GRADCAM-Results"
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


class SpineGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_patch(self, input_tensor):
        use_amp = DEVICE == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = self.model(input_tensor)
            # Use mean of logits as target for backprop
            target = torch.logit(output, eps=1e-6).mean()

        self.model.zero_grad()
        target.backward()

        # Global Average Pooling of Gradients
        pooled_grads = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weighted Activations
        weighted_activations = self.activations * pooled_grads
        
        # Generate Heatmap
        heatmap = torch.sum(weighted_activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Upsample to match input size
        heatmap = F.interpolate(
            heatmap.float(),
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        return output.detach().float(), heatmap.detach().float()


def predict_sliding_window(model, vol, grad_cam):
    d, h, w = vol.shape
    pd, ph, pw = PATCH_SIZE

    prob_map = torch.zeros(vol.shape, device="cpu")
    cam_map = torch.zeros(vol.shape, device="cpu")
    weight_map = torch.zeros(vol.shape, device="cpu")
    patch_window = get_gaussian_window(PATCH_SIZE).cpu()

    stride_d, stride_h, stride_w = [int(p * (1 - OVERLAP)) for p in PATCH_SIZE]
    
    # Steps calculation
    z_steps = sorted(list(set(list(range(0, d - pd + stride_d, stride_d)) + [max(0, d - pd)])))
    y_steps = sorted(list(set(list(range(0, h - ph + stride_h, stride_h)) + [max(0, h - ph)])))
    x_steps = sorted(list(set(list(range(0, w - pw + stride_w, stride_w)) + [max(0, w - pw)])))

    vol_t = torch.from_numpy(vol).float()
    model.eval()

    for z in z_steps:
        for y in y_steps:
            for x in x_steps:
                slice_vol = vol_t[z : z + pd, y : y + ph, x : x + pw]
                curr_d, curr_h, curr_w = slice_vol.shape

                # Padding logic if patch is smaller than PATCH_SIZE (edge cases)
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
                patch.requires_grad = True

                # --- GradCAM Inference ---
                pred_patch, cam_patch = grad_cam.compute_patch(patch)

                pred_patch = pred_patch.squeeze().cpu()
                cam_patch = cam_patch.squeeze().cpu()

                # Apply Gaussian window weighting
                weighted_pred = pred_patch * patch_window
                weighted_cam = cam_patch * patch_window
                weighted_win = patch_window.clone()

                # Remove padding if applied
                if need_pad:
                    weighted_pred = weighted_pred[:curr_d, :curr_h, :curr_w]
                    weighted_cam = weighted_cam[:curr_d, :curr_h, :curr_w]
                    weighted_win = weighted_win[:curr_d, :curr_h, :curr_w]

                # Accumulate results
                prob_map[z : z + curr_d, y : y + curr_h, x : x + curr_w] += weighted_pred
                cam_map[z : z + curr_d, y : y + curr_h, x : x + curr_w] += weighted_cam
                weight_map[z : z + curr_d, y : y + curr_h, x : x + curr_w] += weighted_win

                del patch, pred_patch, cam_patch, weighted_pred, weighted_win, weighted_cam

    weight_map[weight_map == 0] = 1.0
    # Normalize by overlap weights
    return (prob_map / weight_map).numpy(), (cam_map / weight_map).numpy()


def save_visual(ct_vol, pred_mask, cam_vol, subject_id, output_dir):
    # Extract middle slice for visualization
    mid_idx = ct_vol.shape[0] // 2
    ct_slice = ct_vol[mid_idx, :, :].T
    mask_slice = pred_mask[mid_idx, :, :].T
    cam_slice = cam_vol[mid_idx, :, :].T
    binary_mask = (mask_slice > 0.5).astype(np.float32)

    # Normalize heatmap for visualization
    robust_max = np.percentile(cam_vol, 99.5)
    if robust_max > 0:
        cam_slice = cam_slice / robust_max
        cam_slice = np.clip(cam_slice, 0, 1)

    # Threshold low activations for cleaner look
    cam_slice[cam_slice < 0.2] = 0

    fig, ax = plt.subplots(1, 2, figsize=(16, 12))

    # Plot 1: Prediction
    ax[0].imshow(ct_slice, cmap="gray", origin="lower")
    masked_pred = np.ma.masked_where(binary_mask == 0, binary_mask)
    ax[0].imshow(masked_pred, cmap="winter", alpha=0.5, origin="lower")
    ax[0].set_title(
        f"Prediction: {subject_id}", fontsize=14, color="white", backgroundcolor="black"
    )
    ax[0].axis("off")

    # Plot 2: Grad-CAM
    ax[1].imshow(ct_slice, cmap="gray", origin="lower")
    im = ax[1].imshow(cam_slice, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=1)
    ax[1].set_title(
        f"Grad-CAM (Logits): {subject_id}",
        fontsize=14,
        color="white",
        backgroundcolor="black",
    )
    ax[1].axis("off")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"resunet.png")
    plt.savefig(output_path, facecolor="black")
    plt.close()
    print(f"Visualization saved to: {output_path}")


def run_evaluation():
    print(f"--- Loading Model on {DEVICE} ---")
    
    # UPDATED: Instantiate the Context-Off Model
    model = SpineResUNet_cotext_off().to(DEVICE)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Initialize GradCAM on the target layer
    # For SpineResUNet_cotext_off, 'dec1' is the last residual block before final output
    grad_cam = SpineGradCAM(model, model.dec1)

    # Get Subject ID for logging/saving
    file_name = os.path.basename(VOL_PATH)
    subject_id = file_name.split("_")[0]

    print(f"--- Processing Single Instance: {subject_id} ---")
    
    try:
        # Load NIfTI files
        vol_nii = nib.as_closest_canonical(nib.load(VOL_PATH))
        gt_nii = nib.as_closest_canonical(nib.load(MSK_PATH))

        # Preprocessing (Normalization & Clipping)
        vol_data = np.clip(vol_nii.get_fdata(), -1000, 2000)
        vol_data = (vol_data + 1000) / 3000
        gt_data = (gt_nii.get_fdata() > 0).astype(np.float32)

        # Inference with Grad-CAM
        pred_prob, cam_vol = predict_sliding_window(model, vol_data, grad_cam)
        
        # Post-processing
        pred_bin = (pred_prob > 0.5).astype(np.float32)
        pred_bin = keep_largest_blob(pred_bin)

        # Metrics
        dice = compute_dice(pred_bin, gt_data)

        print(f"{subject_id:<30} | Dice: {dice:<12.4f}")

        # Save Visuals
        save_visual(vol_data, pred_bin, cam_vol, subject_id, RESULTS_DIR)

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        gc.collect()


if __name__ == "__main__":
    run_evaluation()
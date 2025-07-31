import numpy as np
import torch
import torch.nn.functional as F

import os 
import matplotlib.pyplot as plt

def load_and_preprocess_npy_square(npy_path_list, target_size=518):
    """
    Load .npy images, normalize to [0,1], center-pad to square.

    Args:
        npy_path_list (List[str]): paths to uint16 npy arrays H×W×3 in [0,16383]
        target_size (int): output height and width (must be >= each image's H,W)

    Returns:
        torch.FloatTensor: shape (N, 3, target_size, target_size), values in [0,1]

    Raises:
        ValueError: if list is empty or any image exceeds target_size.
    """
    if not npy_path_list:
        raise ValueError("At least one .npy path is required")

    imgs = []
    original_coords = []
    max_raw = 2**14 - 1  # 16383

    for p in npy_path_list:
        arr = np.load(p)  # shape H,W,3; dtype=uint16
        h, w, c = arr.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels, got {c} in {p}")
        if h > target_size or w > target_size:
            raise ValueError(f"Image {p} is {h}×{w}, exceeds target_size {target_size}")

        # compute padding amounts
        dh = target_size - h
        dw = target_size - w
        pad = (dw//2, dw - dw//2,  # left, right
               dh//2, dh - dh//2)  # top, bottom

        # coordinates of the original image in the padded output
        x1, y1 = pad[0], pad[2]
        x2, y2 = x1 + w, y1 + h
        original_coords.append([x1, y1, x2, y2, w, h])

        # normalize → float32 → Tensor(C,H,W)
        t = torch.from_numpy(arr.astype(np.float32) / max_raw) \
                 .permute(2, 0, 1)  # C,H,W

        # pad with 1.0 (white)
        t = F.pad(t, pad, mode='constant', value=0.0)
        imgs.append(t)

    # (N,3,target_size,target_size)
    batch = torch.stack(imgs, dim=0)
    coords = torch.tensor(original_coords, dtype=torch.float32)
    return batch, coords


def visualize_npy_sequence(clean_sequence, noisy_sequence, save_dir, step):
    # make output folder
    out_dir = os.path.join(save_dir, "debug", f"step_{step}")
    os.makedirs(out_dir, exist_ok=True)

    # grab only the first batch element and convert to int numpy
    # shape: [T, 3, H, W]
    clean_np = (clean_sequence[0] * 16383).to(torch.int32).cpu().numpy()
    noisy_np = (noisy_sequence[0] * 16383).to(torch.int32).cpu().numpy()

    T = clean_np.shape[0]
    for t in range(T):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        # plot each channel
        for c in range(3):
            # clean
            ax = axes[0, c]
            im = ax.imshow(clean_np[t, c], cmap="plasma",
                           vmin=512, vmax=4000)
            ax.set_title(f"Clean C{c}")
            ax.axis("off")

            # noisy
            ax = axes[1, c]
            im = ax.imshow(noisy_np[t, c], cmap="viridis",
                           vmin=700, vmax=1300)
            ax.set_title(f"Noisy C{c}")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{t:03d}.png"))
        plt.close(fig)
# src/augment_flow.py
# Balances optical flow dataset by augmenting underrepresented classes.
# Reads existing .npy flow files, applies augmentations, saves new ones.
# Run AFTER optical flow preprocessing, BEFORE training.

import os
import numpy as np
import random
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
FLOW_OUTPUT  = r'C:\Dataset\3_OpticalFlow'   # same as config.py
TARGET_COUNT = 80   # how many train samples every class should have
SEED         = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Augmentation functions (all operate on flow arrays: (15, 64, 64, 2)) ─────

def aug_time_shift(flow, shift=2):
    """Shift the flow sequence forward or backward in time."""
    s = random.randint(-shift, shift)
    if s == 0:
        return flow
    result = np.zeros_like(flow)
    if s > 0:
        result[s:] = flow[:-s]
    else:
        result[:s] = flow[-s:]
    return result


def aug_time_stretch(flow, factor=None):
    """Subsample or repeat frames to simulate speed change."""
    T = flow.shape[0]
    if factor is None:
        factor = random.uniform(0.75, 1.25)
    new_len = int(T * factor)
    new_len = max(4, min(new_len, T * 2))
    indices = np.linspace(0, T - 1, new_len).astype(int)
    stretched = flow[indices]
    # Pad or trim back to T frames
    if len(stretched) < T:
        pad = np.zeros((T - len(stretched), *flow.shape[1:]), dtype=flow.dtype)
        stretched = np.concatenate([stretched, pad], axis=0)
    return stretched[:T]


def aug_flip_horizontal(flow):
    """Mirror the sign left-right (flip x-axis of flow vectors)."""
    flipped = flow[:, :, ::-1, :].copy()
    flipped[..., 0] *= -1   # negate horizontal component
    return flipped


def aug_add_noise(flow, sigma=0.02):
    """Add Gaussian noise to flow magnitudes."""
    noise = np.random.normal(0, sigma, flow.shape).astype(np.float32)
    return flow + noise


def aug_scale_magnitude(flow, scale=None):
    """Scale all flow vectors up or down (simulates distance from camera)."""
    if scale is None:
        scale = random.uniform(0.7, 1.3)
    return flow * scale


def aug_spatial_crop(flow, crop_frac=0.85):
    """Simulate a slightly different crop/zoom by cropping then resizing back."""
    import cv2
    T, H, W, C = flow.shape
    ch = int(H * crop_frac)
    cw = int(W * crop_frac)
    top  = random.randint(0, H - ch)
    left = random.randint(0, W - cw)
    result = np.zeros_like(flow)
    for t in range(T):
        for c in range(C):
            cropped = flow[t, top:top+ch, left:left+cw, c]
            result[t, :, :, c] = cv2.resize(cropped, (W, H),
                                             interpolation=cv2.INTER_LINEAR)
    return result


def aug_rotate(flow, angle=None):
    """Rotate spatial dims by a small angle."""
    import cv2
    if angle is None:
        angle = random.uniform(-15, 15)
    T, H, W, C = flow.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    result = np.zeros_like(flow)
    for t in range(T):
        for c in range(C):
            result[t, :, :, c] = cv2.warpAffine(flow[t, :, :, c], M, (W, H))
    return result


# Pool of all augmentations — randomly pick 1-3 per sample
AUGMENTATIONS = [
    aug_time_shift,
    aug_time_stretch,
    aug_flip_horizontal,
    aug_add_noise,
    aug_scale_magnitude,
    aug_spatial_crop,
    aug_rotate,
]


def apply_random_augmentations(flow, n=2):
    """Apply n randomly chosen augmentations in sequence."""
    chosen = random.sample(AUGMENTATIONS, k=min(n, len(AUGMENTATIONS)))
    for fn in chosen:
        flow = fn(flow)
    return flow.astype(np.float32)


# ── Main balancing logic ───────────────────────────────────────────────────────

def balance_split(split='train'):
    base = os.path.join(FLOW_OUTPUT, split)
    if not os.path.isdir(base):
        print(f"[skip] {base} does not exist")
        return

    # Count existing samples
    class_files = {}
    for label in os.listdir(base):
        label_dir = os.path.join(base, label)
        if not os.path.isdir(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        class_files[label] = files

    counts = {k: len(v) for k, v in class_files.items()}
    print(f"\n── {split.upper()} split ──────────────────────────────────")
    for label, count in sorted(counts.items(), key=lambda x: x[1]):
        bar = '█' * min(count, 50)
        print(f"  {label:20s}: {count:4d}  {bar}")

    print(f"\nTarget: {TARGET_COUNT} samples per class")
    generated_total = 0

    for label, files in class_files.items():
        current = len(files)
        needed  = TARGET_COUNT - current

        if needed <= 0:
            print(f"  {label}: already has {current} ✓")
            continue

        print(f"  {label}: generating {needed} augmented samples "
              f"({current} → {TARGET_COUNT})")

        label_dir = os.path.join(base, label)

        for i in range(needed):
            # Pick a random source file to augment
            src_file = random.choice(files)
            src_path = os.path.join(label_dir, src_file)

            try:
                flow = np.load(src_path).astype(np.float32)
            except Exception as e:
                print(f"    [error loading {src_file}] {e}")
                continue

            # Apply 1–3 random augmentations
            n_augs = random.randint(1, 3)
            aug_flow = apply_random_augmentations(flow, n=n_augs)

            # Save with aug_ prefix + index so it won't collide
            save_name = f"aug_{i:04d}_{src_file}"
            np.save(os.path.join(label_dir, save_name), aug_flow)
            generated_total += 1

    print(f"\nGenerated {generated_total} augmented samples for {split} split.")

    # Print updated counts
    print(f"\nUpdated {split} counts:")
    for label in sorted(class_files.keys()):
        label_dir = os.path.join(base, label)
        new_count = len([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        print(f"  {label:20s}: {new_count}")


if __name__ == '__main__':
    balance_split('train')
    # Optionally balance test too (usually not needed):
    # balance_split('test')
    print("\nDone! Now retrain with: python src/train.py")
# src/process_alpha_flows.py
import os
import random
import numpy as np
from preprocess import extract_frames
from hand_crop_flow import compute_flow_with_hand_crop
from config import DYNAMIC_RAW, FLOW_OUTPUT, N_FRAMES
from config import ALPHA_TRAIN_RATIO

def process_alpha_split():
    train_base = os.path.join(FLOW_OUTPUT, 'alpha', 'train')
    test_base  = os.path.join(FLOW_OUTPUT, 'alpha', 'test')

    train_total = 0
    test_total  = 0

    for label in sorted(os.listdir(DYNAMIC_RAW)):
        label_path = os.path.join(DYNAMIC_RAW, label)
        if not os.path.isdir(label_path):
            continue

        all_vids = [v for v in os.listdir(label_path) if v.endswith('.mp4')]
        random.shuffle(all_vids)

        # Cap test first, then cap train from remainder
        split_idx  = int(len(all_vids) * ALPHA_TRAIN_RATIO)
        train_vids = all_vids[:split_idx]
        test_vids  = all_vids[split_idx:]


        # --- Train ---
        save_dir = os.path.join(train_base, label)
        os.makedirs(save_dir, exist_ok=True)
        for vid in train_vids:
            try:
                frames = extract_frames(os.path.join(label_path, vid))
                if len(frames) < 2:
                    continue
                flow = compute_flow_with_hand_crop(frames, target_frames=N_FRAMES)
                np.save(os.path.join(save_dir, vid.replace('.mp4', '.npy')), flow)
                train_total += 1
            except Exception as e:
                print(f"  Skipped {vid}: {e}")

        # --- Test ---
        save_dir = os.path.join(test_base, label)
        os.makedirs(save_dir, exist_ok=True)
        for vid in test_vids:
            try:
                frames = extract_frames(os.path.join(label_path, vid))
                if len(frames) < 2:
                    continue
                flow = compute_flow_with_hand_crop(frames, target_frames=N_FRAMES)
                np.save(os.path.join(save_dir, vid.replace('.mp4', '.npy')), flow)
                test_total += 1
            except Exception as e:
                print(f"  Skipped {vid}: {e}")

    print(f"\nAlpha done. Train: {train_total}, Test: {test_total}")


if __name__ == '__main__':
    random.seed(42)
    process_alpha_split()
# src/process_word_flows.py
import os
import numpy as np
from preprocess import extract_frames
from hand_crop_flow import compute_flow_with_hand_crop
from config import WORD_TRAIN, WORD_TEST, FLOW_OUTPUT, N_FRAMES


def process_word_split(split='train'):
    src = WORD_TRAIN if split == 'train' else WORD_TEST
    out_base = os.path.join(FLOW_OUTPUT, 'word', split)
    count = 0
    skipped = 0

    for label in sorted(os.listdir(src)):
        label_path = os.path.join(src, label)
        if not os.path.isdir(label_path):
            continue

        save_dir = os.path.join(out_base, label)
        os.makedirs(save_dir, exist_ok=True)

        videos = [v for v in os.listdir(label_path) if v.endswith('.mp4')]

        for vid in videos:
            vid_path = os.path.join(label_path, vid)
            try:
                frames = extract_frames(vid_path)
                if len(frames) < 2:
                    print(f"  Skipped (too short): {label}/{vid}")
                    skipped += 1
                    continue
                flow = compute_flow_with_hand_crop(frames, target_frames=N_FRAMES)
                out_path = os.path.join(save_dir, vid.replace('.mp4', '.npy'))
                np.save(out_path, flow)
                count += 1
                print(f"  Saved: {label}/{vid}")
            except Exception as e:
                print(f"  Error {label}/{vid}: {e}")
                skipped += 1

    print(f"\nWord {split}: {count} saved, {skipped} skipped")


if __name__ == '__main__':
    print("Processing WORD TRAIN...")
    process_word_split('train')
    print("\nProcessing WORD TEST...")
    process_word_split('test')
# src/optical_flow.py
import cv2
import numpy as np
import os
from preprocess import extract_frames
from config import WORD_TRAIN, WORD_TEST, FLOW_OUTPUT

def compute_flow(frames, target_frames=15, img_size=(64, 64)):
    flows = []
    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    while len(flows) < target_frames:
        flows.append(np.zeros((*img_size, 2), dtype=np.float32))
    flows = flows[:target_frames]
    result = np.array(flows)
    assert result.shape == (target_frames, *img_size, 2), f"Unexpected shape: {result.shape}"
    return result

def process_dataset(split='train'):
    src = WORD_TRAIN if split == 'train' else WORD_TEST
    out = os.path.join(FLOW_OUTPUT, split)
    count = 0
    for label in os.listdir(src):
        label_path = os.path.join(src, label)
        for vid in os.listdir(label_path):
            if not vid.endswith('.mp4'): continue
            frames = extract_frames(os.path.join(label_path, vid))
            flow   = compute_flow(frames)
            save_dir = os.path.join(out, label)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, vid.replace('.mp4', '.npy')), flow)
            count += 1
            print(f'Processed: {label}/{vid}')
    print(f'Done! {split}: {count} videos processed')

if __name__ == '__main__':
    process_dataset('train')
    process_dataset('test')

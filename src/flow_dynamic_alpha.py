# src/flow_dynamic_alpha.py
import os, numpy as np, random
from optical_flow import compute_flow
from preprocess import extract_frames
from config import FLOW_OUTPUT

DYNAMIC_RAW = r'D:\Rabia_Uni\Semester_4\Programming_for_AI\Projects\Project_2\Dataset\1_UAlpha40_Mendeley\dynamic_signs\raw_videos'
TEST_COUNT = 20  # videos per class to put in test

def process_dynamic_alphabets():
    train_base = os.path.join(FLOW_OUTPUT, 'train')
    test_base  = os.path.join(FLOW_OUTPUT, 'test')
    
    train_total = 0
    test_total  = 0

    for label in os.listdir(DYNAMIC_RAW):
        label_path = os.path.join(DYNAMIC_RAW, label)
        if not os.path.isdir(label_path):
            continue

        # Get all mp4 videos
        all_vids = [v for v in os.listdir(label_path) if v.endswith('.mp4')]
        
        # Shuffle so test selection is random
        random.seed(42)
        random.shuffle(all_vids)

        test_vids  = all_vids[:TEST_COUNT]       # first 20 go to test
        train_vids = all_vids[TEST_COUNT:]        # rest go to train

        print(f'\n{label}: {len(train_vids)} train, {len(test_vids)} test')

        # Process train videos
        save_dir = os.path.join(train_base, label)
        os.makedirs(save_dir, exist_ok=True)
        for vid in train_vids:
            try:
                frames = extract_frames(os.path.join(label_path, vid))
                flow   = compute_flow(frames)
                np.save(os.path.join(save_dir, vid.replace('.mp4', '.npy')), flow)
                train_total += 1
            except Exception as e:
                print(f'  Skipped {vid}: {e}')

        # Process test videos
        save_dir = os.path.join(test_base, label)
        os.makedirs(save_dir, exist_ok=True)
        for vid in test_vids:
            try:
                frames = extract_frames(os.path.join(label_path, vid))
                flow   = compute_flow(frames)
                np.save(os.path.join(save_dir, vid.replace('.mp4', '.npy')), flow)
                test_total += 1
            except Exception as e:
                print(f'  Skipped {vid}: {e}')

    print(f'\nDone! Train: {train_total} videos, Test: {test_total} videos')

if __name__ == '__main__':
    process_dynamic_alphabets()
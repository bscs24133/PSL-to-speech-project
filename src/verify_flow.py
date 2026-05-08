# src/verify_flow.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from config import FLOW_OUTPUT

def visualize_flow(flow_array, title='Optical Flow'):
    flow = flow_array[0]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig('../outputs/flow_sample.png')
    plt.show()
    print('Saved to outputs/flow_sample.png')

if __name__ == '__main__':
    npy_files = glob.glob(FLOW_OUTPUT + '/train/**/*.npy', recursive=True)
    if npy_files:
        flow = np.load(npy_files[0])
        label = os.path.basename(os.path.dirname(npy_files[0]))
        print(f'Visualizing flow for: {label}')
        print(f'Flow shape: {flow.shape}')
        visualize_flow(flow, title=f'Optical Flow - {label}')
    else:
        print('No .npy files found. Run optical_flow.py first.')

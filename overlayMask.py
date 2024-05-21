import cv2
import numpy as np
import matplotlib.pyplot as plt

mask_path = 'dataset\pngs\AffinedPng\s005_mask_slice_1.png'
brain_path = 'dataset\pngs\RawPng\s005_slice_1.png'

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
brain = cv2.imread(brain_path, cv2.IMREAD_GRAYSCALE)

if mask.shape != brain.shape:
    raise ValueError("大小不一致")

_, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

brain_color = cv2.cvtColor(brain, cv2.COLOR_GRAY2BGR)

color_mask = np.zeros_like(brain_color)
color_mask[binary_mask == 255] = [0, 255, 0]  #绿色

overlay = cv2.addWeighted(brain_color, 1, color_mask, 0.5, 0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Brain Image')
plt.imshow(brain, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Mask Image')
plt.imshow(mask, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Overlay Image')
plt.imshow(overlay)

plt.show()

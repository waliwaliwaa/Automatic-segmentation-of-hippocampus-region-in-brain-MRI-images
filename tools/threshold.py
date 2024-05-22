import cv2
import numpy as np
import matplotlib.pyplot as plt

brain_path = 'dataset\pngs\RawPng\s005_slice_1.png'
brain = cv2.imread(brain_path, cv2.IMREAD_GRAYSCALE)

# 手动选择海马体所在的ROI
x, y, w, h = 150, 100, 200, 200  # 海马体位置
roi = brain[y:y+h, x:x+w]

# 直方图分析ROI区域
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 5))
plt.title('Histogram of ROI')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(hist)
plt.show()

# 基于ROI区域的直方图，设定一个合适的阈值，逻辑不是很对

min_val = np.min(roi)
max_val = np.max(roi)
print(f'Min pixel value in ROI: {min_val}')
print(f'Max pixel value in ROI: {max_val}')

_, thresholded_image = cv2.threshold(brain, min_val, max_val, cv2.THRESH_BINARY)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Brain Image')
plt.imshow(brain, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('ROI Image')
plt.imshow(roi, cmap='gray')

plt.subplot(1, 3, 3)
plt.title(f'Thresholded Image (Threshold: {min_val} to {max_val})')
plt.imshow(thresholded_image, cmap='gray')

plt.show()

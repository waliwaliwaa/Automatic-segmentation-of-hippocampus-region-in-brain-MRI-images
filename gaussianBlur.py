import os
import cv2

input_dir = 'dataset/pngs/RawPng'
output_dir = 'dataset/pngs/RawGauss'


kernel_size = (5, 5)
sigma = 1.0


for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path, 0)

        filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)

        cv2.imwrite(output_path, filtered_image)

        print(f'Processed and saved: {output_path}')

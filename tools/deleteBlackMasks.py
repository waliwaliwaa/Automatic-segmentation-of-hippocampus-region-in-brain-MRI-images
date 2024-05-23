import cv2
import os

def delete_black_masks(mask_dir, image_dir):
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(mask_dir, filename)
            
            image_name = filename[0:4] + filename[9:]
            # print("mask: " + filename + "img: " + image_name)
            image_path = os.path.join(image_dir, image_name)  
            # print("mask: " + mask_path + "img: " + image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if cv2.countNonZero(mask) == 0:  # countNonZero函数返回非零像素的数量
                os.remove(mask_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
                print(f"Deleted {mask_path} and {image_path}")

mask_dir = 'dataset/pngs/MaskAll/'  
image_dir = 'dataset/pngs/RawAll/'  
delete_black_masks(mask_dir, image_dir)

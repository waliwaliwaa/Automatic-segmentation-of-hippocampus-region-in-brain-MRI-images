import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = 'dataset'
image_dir = "dataset/pngs/RawGauss"
mask_dir = "dataset/pngs/AffinedPng"

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

assert len(image_files) == len(mask_files)
for img_file, mask_file in zip(image_files, mask_files):
    assert img_file.split('.')[0][1:4] == mask_file.split('.')[0][1:4] and img_file.split('.')[0][-1] == mask_file.split('.')[0][-1]

train_images, val_test_images, train_masks, val_test_masks = train_test_split(
    image_files, mask_files, test_size=0.4, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(
    val_test_images, val_test_masks, test_size=0.5, random_state=42)

output_dirs = {
    'train': {
        'images': "dataset_train/images",
        'masks': "dataset_train/masks"
    },
    'val': {
        'images': "dataset_val/images",
        'masks': "dataset_val/masks"
    },
    'test': {
        'images': "dataset_test/images",
        'masks': "dataset_test/masks"
    }
}


def copy_files(file_list, src_dir, dst_dir):
    for file_name in file_list:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy(src_path, dst_path)


copy_files(train_images, image_dir, output_dirs['train']['images'])
copy_files(train_masks, mask_dir, output_dirs['train']['masks'])
copy_files(val_images, image_dir, output_dirs['val']['images'])
copy_files(val_masks, mask_dir, output_dirs['val']['masks'])
copy_files(test_images, image_dir, output_dirs['test']['images'])
copy_files(test_masks, mask_dir, output_dirs['test']['masks'])

print("Done")

import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def save_slices_as_images(data, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    slice_0 = data[data.shape[0] // 2, :, :]  # Middle slice in the first dimension
    slice_1 = data[:, data.shape[1] // 2, :]  # Middle slice in the second dimension
    slice_2 = data[:, :, data.shape[2] // 2]  # Middle slice in the third dimension

    slices = [slice_0, slice_1, slice_2]
    slice_names = ["slice_0", "slice_1", "slice_2"]

    for slice_data, name in zip(slices, slice_names):
        plt.imshow(slice_data.T, cmap="gray", origin="lower")
        output_path = os.path.join(output_dir, f"{prefix}_{name}.png")
        plt.axis('off')  # Turn off the axis
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {output_path}")

def convert_nii_to_images(nii_file, output_dir):
    img = nib.load(nii_file)
    data = img.get_fdata()

    prefix = os.path.splitext(os.path.basename(nii_file))[0]
    save_slices_as_images(data, output_dir, prefix)

if __name__ == "__main__":
    # nii_directory = "dataset\dataset\RawNii"  # 修改为存放.nii文件的目录路径
    nii_directory = "dataset\dataset\AffinedNii"
    # output_directory = "dataset\dataset\RawPng"  # 修改为保存图像文件的目录路径
    output_directory = "dataset\dataset\AffinedPng"

    for filename in os.listdir(nii_directory):
        if filename.endswith(".nii"):
            nii_file_path = os.path.join(nii_directory, filename)
            convert_nii_to_images(nii_file_path, output_directory)

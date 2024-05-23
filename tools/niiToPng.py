import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def save_slices_as_images(data, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    # slice_0 = data[data.shape[0] // 3, :, :] 
    # slice_1 = data[:, data.shape[1] // 3, :]
    # slice_2 = data[:, :, data.shape[2] // 3] 

    # slices = [slice_0, slice_1, slice_2]
    # slice_names = ["slice_0", "slice_1", "slice_2"]
    # print(data.shape)
    # for slice_data, name in zip(slices, slice_names):
    #     plt.imshow(slice_data.T, cmap="gray", origin="lower")
    #     output_path = os.path.join(output_dir, f"{prefix}_{name}.png")
    #     plt.axis('off')
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #     plt.close()
    #     print(f"Saved {output_path}")
    dimensions = ['slice_dim0_', 'slice_dim1_', 'slice_dim2_']
    for dim_index, dim_name in enumerate(dimensions):
        # 遍历当前维度的所有切片
        for i in range(data.shape[dim_index]):
            if dim_index == 0:
                slice_data = data[i, :, :]
            elif dim_index == 1:
                slice_data = data[:, i, :]
            else:
                slice_data = data[:, :, i]

            output_path = os.path.join(output_dir, f"{prefix}_{dim_name}{i}.png")
            plt.imshow(slice_data.T, cmap="gray", origin="lower")
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved {output_path}")

def convert_nii_to_images(nii_file, output_dir):
    img = nib.load(nii_file)
    data = img.get_fdata()

    prefix = os.path.splitext(os.path.basename(nii_file))[0]
    save_slices_as_images(data, output_dir, prefix)

if __name__ == "__main__":
    # nii_directory = "dataset\dataset\RawNii" 
    nii_directory = "AffinedNii"
    # output_directory = "dataset\dataset\RawPng" 
    output_directory = "dataset/pngs/MaskAll"

    for filename in os.listdir(nii_directory):
        if filename.endswith(".nii"):
            nii_file_path = os.path.join(nii_directory, filename)
            convert_nii_to_images(nii_file_path, output_directory)

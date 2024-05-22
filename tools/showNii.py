import nibabel as nib
import matplotlib.pyplot as plt



def display_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

def main(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Select slices to display
    slice_0 = data[data.shape[0] // 2, :, :] 
    slice_1 = data[:, data.shape[1] // 2, :]
    slice_2 = data[:, :, data.shape[2] // 2] 

    # Display slices
    display_slices([slice_0, slice_1, slice_2])

if __name__ == "__main__":
    file_path = "dataset\dataset\AffinedManualSegImageNIfTI\s001_mask.nii" 
    main(file_path)
import gzip
import os
import shutil

def unGz(inPath: str, outPath: str): 
    with gzip.open(inPath, 'rb') as f_in:
        with open(outPath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# unGz("dataset\dataset\AffinedManualSegImageNIfTI\s001_mask.nii.gz")

# inDirectory = "dataset\dataset\RawImageNIfTI"
inDirectory = "dataset\dataset\AffinedManualSegImageNIfTI"
# outDirectory = "dataset\dataset\RawNii"
outDirectory = "dataset\dataset\AffinedNii"
for filename in os.listdir(inDirectory):
    if filename.endswith(".gz"):
        file_path = os.path.join(inDirectory, filename)
        output_path = os.path.join(outDirectory, filename.removesuffix(".gz"))
        unGz(file_path, output_path)
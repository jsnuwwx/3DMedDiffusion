import nibabel as nib
import numpy as np
from PIL import Image
import os

def main():
    input_nii = 'test_pre_op_decode_PERFECT.nii.gz'
    output_png = 'preview_result.png'
    
    if not os.path.exists(input_nii):
        print(f"Error: {input_nii} not found.")
        return

    # Load NIfTI
    img = nib.load(input_nii)
    data = img.get_fdata()
    print(f"Volume shape: {data.shape}")
    
    # Extract middle axial slice (dim 0 based on our transpose)
    # Based on (192, 96, 96) - 192 is the SI axis
    slice_idx = data.shape[0] // 2
    slice_data = data[slice_idx, :, :]
    
    # Normalize to 0-255 for PNG
    d_min = np.min(slice_data)
    d_max = np.max(slice_data)
    print(f"Slice range: {d_min:.2f} to {d_max:.2f}")
    
    if d_max - d_min > 1e-5:
        slice_norm = (slice_data - d_min) / (d_max - d_min) * 255.0
    else:
        slice_norm = slice_data * 0
        
    slice_norm = slice_norm.astype(np.uint8)
    
    # Save as PNG
    img_pil = Image.fromarray(slice_norm)
    img_pil.save(output_png)
    print(f"Saved preview to {output_png}")

if __name__ == "__main__":
    main()

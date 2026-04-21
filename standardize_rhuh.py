import os
import SimpleITK as sitk
import shutil
import tempfile
from tqdm import tqdm
from pathlib import Path

def safe_read_image(img_path):
    """
    Copy a file to a temporary ASCII path to avoid SimpleITK's 
    encoding issues on Windows with Chinese characters.
    """
    suffix = "".join(Path(img_path).suffixes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        temp_name = tf.name
    
    # Copy original to temp
    shutil.copy(img_path, temp_name)
    
    # Read from temp
    try:
        img = sitk.ReadImage(temp_name)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return img

def safe_write_image(image, output_path):
    """
    Write to a temp file then move to final destination.
    """
    suffix = "".join(Path(output_path).suffixes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        temp_name = tf.name
    
    try:
        sitk.WriteImage(image, temp_name)
        # Ensure target dir exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(temp_name, output_path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def standardize_image(img_path, ref_img, output_path):
    """
    Standardize a source image to match a reference image's grid, orientation, and spacing.
    Uses CenteredTransformInitializer to align physical centers.
    """
    source_img = safe_read_image(img_path)
    
    # 🧠 Key Fix: Align physical centers of the two images
    # This prevents "empty images" when origins are wildly different.
    initial_transform = sitk.CenteredTransformInitializer(
        ref_img, 
        source_img, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # Define Resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(initial_transform) # Apply centering
    
    # Execute Resample
    standardized_img = resampler.Execute(source_img)
    
    # Save safely
    safe_write_image(standardized_img, output_path)

def main():
    # --- Paths ---
    # We use string here but keep it raw for robustness
    DATA_ROOT = Path(r"E:\deep learning code\3D_MedDiffusion_Code\RHUH-GBM_nii_v1 - 副本")
    REF_PATH = Path(r"E:\deep learning code\3D_MedDiffusion_Code\BraTsdataset\data\BraTS-GLI-00005-100-t1c.nii")
    OUTPUT_ROOT = Path(r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_standardized")
    
    # Load Reference safely
    if not REF_PATH.exists():
        print(f"❌ Error: Reference image not found at {REF_PATH}")
        return
    ref_img = safe_read_image(str(REF_PATH))
    print(f"📖 Reference Loaded: {REF_PATH} (Shape: {ref_img.GetSize()})")

    # Patient Folders
    patient_dirs = [d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()]
    print(f"🔍 Found {len(patient_dirs)} patients.")

    for p_id in tqdm(patient_dirs):
        p_path = DATA_ROOT / p_id
        
        # --- CLEANUP LOGIC ---
        # User requested to delete 0_latents and 1_latents folders
        for folder_name in ["0_latents", "1_latents"]:
            latent_folder = p_path / folder_name
            if latent_folder.exists():
                try:
                    shutil.rmtree(latent_folder)
                except Exception as e:
                    print(f"⚠️ Could not delete {latent_folder}: {e}")

        # --- STANDARDIZATION LOGIC ---
        # Pre (0) and Post (1)
        pre_dir = p_path / "0"
        post_dir = p_path / "1"
        
        if not pre_dir.exists() or not post_dir.exists():
            continue
            
        # Find t1ce files
        pre_t1ce = list(pre_dir.glob("*_t1ce.nii.gz"))
        post_t1ce = list(post_dir.glob("*_t1ce.nii.gz"))
        
        if pre_t1ce and post_t1ce:
            out_pre = str(OUTPUT_ROOT / f"{p_id}-100-t1c.nii.gz")
            out_post = str(OUTPUT_ROOT / f"{p_id}-101-t1c.nii.gz")
            
            try:
                standardize_image(str(pre_t1ce[0]), ref_img, out_pre)
                standardize_image(str(post_t1ce[0]), ref_img, out_post)
            except Exception as e:
                print(f"❌ Error processing {p_id}: {e}")

    print(f"\n✅ All done! Standardized images are in {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()

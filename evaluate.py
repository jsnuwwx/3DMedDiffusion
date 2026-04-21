import os
import json
import glob
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def normalize(data):
    """Normalize data to [0, 1] range."""
    dmin = data.min()
    dmax = data.max()
    if dmax - dmin > 0:
        return (data - dmin) / (dmax - dmin)
    return data

def main():
    # --- Configuration ---
    # Path to the folder containing your generated .nii.gz files
    PRED_DIR = r"./batch_predictions_PERFECT"
    # Path to the dataset JSON for pairing
    JSON_PATH = r"BraTsdataset/brats_2024_pairs.json"
    # Local root to remap JSON paths if needed
    LOCAL_DATA_ROOT = r"E:\deep learning code\3D_MedDiffusion_Code\BraTsdataset\data"
    
    # --- Init ---
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: Cannot find JSON file at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        pair_list = json.load(f)

    # Search for predicted files
    pred_files = glob.glob(os.path.join(PRED_DIR, "*_predicted.nii.gz"))
    if not pred_files:
        print(f"❌ No predicted files found in {PRED_DIR}")
        return

    print(f"🔍 Found {len(pred_files)} predictions to evaluate.\n" + "="*50)

    all_ssim = []
    all_psnr = []
    all_mae = []

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)
        # Extract patient ID and sequence from filename
        # Example: BraTS-GLI-00005-100-t1c_predicted.nii.gz
        base_id = pred_name.split("_predicted")[0] # BraTS-GLI-00005-100-t1c
        patient_tag = "-".join(base_id.split("-")[:3]) # BraTS-GLI-00005
        
        print(f"▶️ Evaluating: {base_id}")

        # Find corresponding GT (post-op) in JSON
        gt_path = None
        for pair in pair_list:
            if patient_tag in pair['post_op_img']:
                # Remap remote path to local path
                remote_filename = os.path.basename(pair['post_op_img'])
                gt_path = os.path.join(LOCAL_DATA_ROOT, remote_filename)
                break
        
        if not gt_path or not os.path.exists(gt_path):
            print(f"   ⚠️ Ground Truth not found for {patient_tag}. Skipping...")
            continue

        try:
            # Load Prediction
            # 182x218x182 (as saved in inference.py)
            pred_img = nib.load(pred_path).get_fdata()
            pred_norm = normalize(pred_img)

            # Load Ground Truth
            gt_nii = nib.load(gt_path)
            gt_img = gt_nii.get_fdata()
            
            # Spatial Alignment: GT needs to match Prediction shape (182, 218, 182)
            # Prediction from inference.py underwent: resample -> flip -> flip
            # We must do the same to GT for 1:1 comparison
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).float()
            gt_resized = F.interpolate(gt_tensor, size=(182, 218, 182), mode='trilinear', align_corners=False)
            gt_final = gt_resized.squeeze().numpy()
            
            # Apply the same flips as in inference.py to match spatial orientation
            gt_final = np.flip(gt_final, axis=2)
            gt_final = np.flip(gt_final, axis=1)
            gt_final = normalize(gt_final)

            # --- Strict Masked Metrics ---
            # Create a mask of the brain (where GT intensity is non-trivial)
            # This prevents the black background from inflating the scores.
            mask = gt_final > 0.05 
            
            # Find the bounding box of the brain to crop and calculate SSIM fairly
            coords = np.array(np.nonzero(mask))
            if coords.size > 0:
                min_c = coords.min(axis=1)
                max_c = coords.max(axis=1)
                pred_crop = pred_norm[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
                gt_crop = gt_final[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
                
                # Calculate SSIM only on the cropped brain region
                val_ssim = ssim(gt_crop, pred_crop, data_range=1.0)
                
                # Calculate MAE and PSNR only on masked voxels
                diff = (gt_final[mask] - pred_norm[mask])
                val_mae = np.mean(np.abs(diff))
                mse = np.mean(diff**2)
                val_psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
            else:
                val_ssim, val_psnr, val_mae = 0, 0, 1

            all_ssim.append(val_ssim)
            all_psnr.append(val_psnr)
            all_mae.append(val_mae)

            print(f"   📊 [STRICT] SSIM: {val_ssim:.4f} | PSNR: {val_psnr:.2f} | MAE: {val_mae:.4f}")

        except Exception as e:
            print(f"   ❌ Error processing {base_id}: {str(e)}")

    # --- Summary ---
    if all_ssim:
        print("\n" + "="*50)
        print("🏆 FINAL QUANTITATIVE RESULTS (Averages)")
        print(f"✨ Mean SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"✨ Mean PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
        print(f"✨ Mean MAE:  {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        print("="*50)
    else:
        print("\n❌ No successful evaluations to report.")

if __name__ == "__main__":
    main()

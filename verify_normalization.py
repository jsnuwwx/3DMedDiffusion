import torch
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "BraTsdataset"))

from BraTsdataset.brats_dataset import get_brats_dataloader
from dataset.Singleres_dataset import Singleres_dataset

def verify():
    json_path = "BraTsdataset/brats_2024_pairs.json"
    
    print("--- Testing BraTsDataset (Stage 1/2) ---")
    loader = get_brats_dataloader(json_path, batch_size=1)
    for batch in loader:
        data = batch["data"]
        print(f"Shape: {data.shape}")
        print(f"Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"Mean: {data.mean():.4f}")
        break

    print("\n--- Testing SingleResDataset (Stage 3 / Latent Gen) ---")
    # Test Latent Generation Mode (A)
    ds_gen = Singleres_dataset(json_path, generate_latents=True)
    img, _ = ds_gen[0]
    print(f"[Latent Gen Mode] Range: [{img.min():.4f}, {img.max():.4f}]")

    # Test Training Mode (B) - Note: this loads .nii latents, not raw images
    # We only need to ensure Mode A is correct for feature extraction.
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify()

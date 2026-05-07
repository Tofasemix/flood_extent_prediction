import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_unet import MultimodalFloodModel

def run_inference():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Initialize Model and Load Weights
    # Must match the exact architecture used during training
    model = MultimodalFloodModel(tabular_dim=3, bottleneck_dim=512).to(device)
    
    # Load the state dictionary (the learned weights)
    weights_path = Path(__file__).resolve().parent / "multimodal_flood_unet.pth"
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    
    # CRITICAL: Put the model in evaluation mode. 
    # This disables Dropout and freezes BatchNorm statistics.
    model.eval()

    # 3. Load the Dataset (Without Augmentations)
    # We want to see the pure, unrotated physical data
    dataset = CycloneFloodDataset(csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), img_dir=str(ROOT_DIR / 'Dataset/'), transform=None)
    
    # Pick a random cyclone from the dataset
    sample_idx = random.randint(0, len(dataset) - 1)
    spatial_tensor, tabular_tensor, ground_truth_mask = dataset[sample_idx]

    # Add batch dimension [1, C, H, W] and move to GPU
    spatial_input = spatial_tensor.unsqueeze(0).to(device)
    tabular_input = tabular_tensor.unsqueeze(0).to(device)

   # 4. The Forward Pass
    with torch.no_grad(): 
        raw_logits = model(spatial_input, tabular_input)
        probabilities = torch.sigmoid(raw_logits)
        
        # --- NEW DIAGNOSTIC PRINTS ---
        print(f"Max raw logit: {raw_logits.max().item():.4f}")
        print(f"Max probability: {probabilities.max().item():.6f}")
        print(f"Mean probability: {probabilities.mean().item():.6f}")
        
        # Drop the threshold drastically to see if ANY signal is hiding
        predicted_mask = (probabilities > 0.5).float()

    # 5. Data Extraction for Plotting
    # Move tensors back to CPU and convert to standard numpy arrays
    tp36_img = spatial_tensor[0].cpu().numpy()  # Channel 0: Precipitation
    slope_img = spatial_tensor[1].cpu().numpy() # Channel 1: Topography
    true_mask = ground_truth_mask[0].cpu().numpy()
    pred_mask = predicted_mask.squeeze().cpu().numpy()

    # 6. Generate the Academic Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.suptitle(f"Cyclone Inference Results (Sample ID: {sample_idx})", fontsize=16)

    # Topography
    axes[0].imshow(slope_img, cmap='terrain')
    axes[0].set_title('Topography (SLOPE)')
    axes[0].axis('off')

    # Precipitation
    axes[1].imshow(tp36_img, cmap='Blues')
    axes[1].set_title('Precipitation (TP36)')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(true_mask, cmap='gray')
    axes[2].set_title('Ground Truth Flood Mask')
    axes[2].axis('off')

    # Model Prediction
    axes[3].imshow(pred_mask, cmap='magma')
    axes[3].set_title('U-Net Prediction')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()
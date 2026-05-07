import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

class CycloneFloodDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Load the CSV
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Tabular Data Selection
        tabular_cols = ['pressure', 'vmax', 'wind']
        self.tabular_data = self.data_frame[tabular_cols].values
        
        # Calculate mean and std for Z-score normalization
        self.tab_mean = self.tabular_data.mean(axis=0)
        self.tab_std = self.tabular_data.std(axis=0)
        
        # Normalize the tabular data immediately to prevent dominating weights
        self.tabular_data = (self.tabular_data - self.tab_mean) / (self.tab_std + 1e-8)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the cyclone ID (e.g., WILLA_2018-10-23)
        cyclone_id = self.data_frame.iloc[idx, 0]

        # 2. Construct Exact File Paths based on our Audit
        # Note the PRECIPITATION folder containing TP36_ files
        tp36_path = os.path.join(self.img_dir, 'PRECIPITATION', f'TP36_{cyclone_id}.png')
        slope_path = os.path.join(self.img_dir, 'SLOPE', f'SLOPE_{cyclone_id}.png')
        flood_path = os.path.join(self.img_dir, 'FLOOD', f'FLOOD_{cyclone_id}.png')

        # 3. Load Images (cv2.IMREAD_GRAYSCALE ensures they load as 2D 512x512 arrays)
        tp36_img = cv2.imread(tp36_path, cv2.IMREAD_GRAYSCALE)
        slope_img = cv2.imread(slope_path, cv2.IMREAD_GRAYSCALE)
        flood_img = cv2.imread(flood_path, cv2.IMREAD_GRAYSCALE)

        # 4. Stack Spatial Inputs
        # Stack TP36 and SLOPE to create a (512, 512, 2) numpy array
        spatial_input = np.stack([tp36_img, slope_img], axis=-1)

        # 5. Apply Spatial Augmentations (Albumentations)
        # This randomly flips/rotates both the inputs and the target mask in perfect alignment
        if self.transform is not None:
            augmented = self.transform(image=spatial_input, mask=flood_img)
            spatial_input = augmented['image']
            flood_img = augmented['mask']
        
        # 6. Final Preprocessing
        # Normalize spatial inputs to [0, 1] range
        spatial_input = spatial_input.astype(np.float32) / 255.0
        
        # Binarize flood mask strictly to 0.0 and 1.0
        flood_mask = (flood_img > 0).astype(np.float32)
        
        # Expand dims so the mask has a channel dimension: (512, 512, 1)
        flood_mask = np.expand_dims(flood_mask, axis=-1)

        # 7. Convert to PyTorch Tensors
        # PyTorch expects channels first: (C, H, W) instead of (H, W, C)
        spatial_tensor = torch.from_numpy(spatial_input).permute(2, 0, 1)
        flood_tensor = torch.from_numpy(flood_mask).permute(2, 0, 1)
        
        # Extract Tabular Tensor (Size: 3)
        tabular_tensor = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        return spatial_tensor, tabular_tensor, flood_tensor

# --- Quick Architecture Integration Test ---
if __name__ == "__main__":
    # Create a dummy transform for testing (optional)
    import albumentations as A
    test_transform = A.Compose([A.HorizontalFlip(p=1.0)])
    
    # Instantiate the dataset
    dataset = CycloneFloodDataset(
        csv_file='clean_tabular_data.csv', 
        img_dir='Dataset/', 
        transform=test_transform
    )
    
    # Pull the first tuple
    spatial, tabular, mask = dataset[0]
    
    print(f"Spatial shape: {spatial.shape}")  
    print(f"Tabular shape: {tabular.shape}")  
    print(f"Mask shape: {mask.shape}")
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class MultimodalFloodModel(nn.Module):
    def __init__(self, tabular_dim=3, bottleneck_dim=512):
        super(MultimodalFloodModel, self).__init__()
        
        # 1. Visual Backbone (Spatial Data)
        # in_channels=2 specifically for the stacked TP36 and SLOPE images.
        # We use a ResNet34 encoder; it is lightweight, fast, and highly effective.
        self.unet = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights="imagenet", 
            in_channels=2, 
            classes=1 # Outputting a single binarized flood mask
        )

        # 2. Tabular Branch (1D Cyclone Data: Pressure, Vmax, Wind)
        # Projects the 3 variables into a dense representation matching the U-Net bottleneck.
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )

    def forward(self, x_spatial, x_tabular):
        # A. Extract spatial features via the U-Net Encoder
        # We wrap this in a list() to ensure it's fully mutable
        features = list(self.unet.encoder(x_spatial))

        # B. Process tabular data into a dense vector
        tab_embedding = self.mlp(x_tabular) # Shape: (Batch, bottleneck_dim)

        # C. Bottleneck Fusion
        bottleneck = features[-1]
        batch_size, channels, h, w = bottleneck.size()

        # Reshape and expand the 1D tabular vector to match the 2D spatial grid
        tab_expanded = tab_embedding.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)

        # Fuse by adding the expanded tabular data directly into the spatial feature map
        features[-1] = bottleneck + tab_expanded 

        # D. Decode the fused features
        # We use a try-except block to make the code bulletproof across all library versions
        try:
            # Latest SMP versions expect a single list
            decoder_output = self.unet.decoder(features)
        except TypeError:
            # Older SMP versions expect unpacked arguments
            decoder_output = self.unet.decoder(*features)

        segmentation_mask = self.unet.segmentation_head(decoder_output)

        return segmentation_mask

# --- Hardware & Dimension Test ---
if __name__ == "__main__":
    # Mocking a batch of 8 images, 2 channels (TP36, SLOPE), 512x512 resolution
    dummy_images = torch.randn(8, 2, 512, 512).cuda()
    
    # Mocking a batch of 8 tabular tuples (Pressure, Vmax, Wind)
    dummy_tabular = torch.randn(8, 3).cuda()
    
    # Initialize model and push to your RTX 4500
    model = MultimodalFloodModel().cuda()
    
    # Forward pass
    output = model(dummy_images, dummy_tabular)
    
    print(f"Spatial Input Shape: {dummy_images.shape}")
    print(f"Tabular Input Shape: {dummy_tabular.shape}")
    print(f"Output Flood Mask Shape: {output.shape}") 
    # Expected Output: torch.Size([8, 1, 512, 512])
    print("✅ Forward pass successful on GPU!")
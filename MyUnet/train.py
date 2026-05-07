import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import sys
import csv
from pathlib import Path

# --- RESOLUCIÓN DINÁMICA DE RUTAS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_unet import MultimodalFloodModel

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Nota: Sin ToTensorV2() aquí porque CycloneFloodDataset 
        # hace el .astype(np.float32) manualmente en sus entrañas.
    ]
    return A.Compose(train_transform)

def train_model():
    # 1. Hyperparameters & Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} (RTX 4500 expected)")
    
    epochs = 150
    batch_size = 16  
    learning_rate = 1e-4

    # --- CONFIGURACIÓN DEL LOGGER ---
    log_path = Path(__file__).resolve().parent / "unet_training_logs.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'lr']) # Encabezados del CSV
    print(f"Los logs de entrenamiento se guardarán automáticamente en: {log_path}")

    # 2. Load Data
    dataset = CycloneFloodDataset(
        csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), 
        img_dir=str(ROOT_DIR / 'Dataset/'), 
        transform=get_training_augmentation()
    )
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 3. Initialize Model
    model = MultimodalFloodModel(tabular_dim=3, bottleneck_dim=512).to(device)

    # 4. Define Specialized Loss and Optimizer
    dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    pos_weight = torch.tensor([50.0]).to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def combined_loss(y_pred, y_true):
        return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. The Training Loop
    print(f"Starting training for {epochs} epochs on {len(dataset)} cyclones...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (spatial_inputs, tabular_inputs, flood_masks) in enumerate(train_loader):
            spatial_inputs = spatial_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)
            
            # Protección Dimensional para el BCE Loss
            flood_masks = flood_masks.to(device)
            if len(flood_masks.shape) == 3:
                flood_masks = flood_masks.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(spatial_inputs, tabular_inputs)
            loss = combined_loss(predictions, flood_masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Extraer métricas de la época
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Guardar silenciosamente en el CSV
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, current_lr])
        
        scheduler.step(avg_loss)

    print("\nTraining complete! Saving model weights...")
    save_path = str(Path(__file__).resolve().parent / "multimodal_flood_unet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved exactly at: {save_path}")

if __name__ == "__main__":
    train_model()
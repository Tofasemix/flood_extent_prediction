import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import sys
import csv
import json
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
    encoder_weights = "imagenet"

    # --- CONFIGURACIÓN DEL LOGGER ---
    log_path = Path(__file__).resolve().parent / "unet_training_logs.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    print(f"Los logs de entrenamiento se guardarán automáticamente en: {log_path}")

    # --- CONFIGURACIÓN EXPLÍCITA DEL EXPERIMENTO ---
    print("\nConfiguración del experimento:")
    print("  Architecture          : Multimodal U-Net")
    print("  Encoder               : ResNet-34")
    print(f"  Encoder initialization: {encoder_weights}")
    print("  Spatial channels      : 2")
    print("  Tabular features      : 3")

    run_config = {
        "architecture": "Multimodal U-Net",
        "encoder": "resnet34",
        "encoder_weights": encoder_weights,
        "spatial_channels": 2,
        "tabular_dim": 3,
        "bottleneck_dim": 512,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
    }

    config_path = Path(__file__).resolve().parent / "unet_run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=4)
    print(f"  Run config saved to   : {config_path}\n")

    # 2. Load Train / Validation Data
    train_csv = ROOT_DIR / "splits" / "train.csv"
    val_csv = ROOT_DIR / "splits" / "val.csv"

    train_dataset = CycloneFloodDataset(
        csv_file=str(train_csv),
        img_dir=str(ROOT_DIR / "Dataset"),
        transform=get_training_augmentation(),
    )

    # Fit normalization statistics ONLY on the training split.
    train_mean, train_std = train_dataset.get_tabular_stats()

    val_dataset = CycloneFloodDataset(
        csv_file=str(val_csv),
        img_dir=str(ROOT_DIR / "Dataset"),
        transform=None,
        tabular_mean=train_mean,
        tabular_std=train_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")
    print(f"Tabular feature order: {train_dataset.tabular_cols}")
    print(f"Train tabular mean : {train_mean}")
    print(f"Train tabular std  : {train_std}")

    # Add the exact split/normalization protocol to the run config.
    run_config.update({
        "train_csv": str(train_csv.relative_to(ROOT_DIR)),
        "val_csv": str(val_csv.relative_to(ROOT_DIR)),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "tabular_feature_order": train_dataset.tabular_cols,
        "tabular_mean": train_mean.tolist(),
        "tabular_std": train_std.tolist(),
        "normalization_fit_split": "train",
    })

    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=4)

    # 3. Initialize Model
    model = MultimodalFloodModel(
        tabular_dim=3,
        bottleneck_dim=512,
        encoder_weights=encoder_weights,
    ).to(device)

    # 4. Define Specialized Loss and Optimizer
    dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    pos_weight = torch.tensor([50.0]).to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def combined_loss(y_pred, y_true):
        return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. Training / Validation Loop
    print(
        f"Starting training for {epochs} epochs "
        f"on {len(train_dataset)} training cyclones..."
    )

    best_val_loss = float("inf")
    best_epoch = 0
    save_path = (
        Path(__file__).resolve().parent
        / "multimodal_flood_unet_best_split.pth"
    )

    for epoch in range(epochs):
        # -------------------------
        # Training phase
        # -------------------------
        model.train()
        train_loss_sum = 0.0

        for spatial_inputs, tabular_inputs, flood_masks in train_loader:
            spatial_inputs = spatial_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)

            flood_masks = flood_masks.to(device)
            if len(flood_masks.shape) == 3:
                flood_masks = flood_masks.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(spatial_inputs, tabular_inputs)
            loss = combined_loss(predictions, flood_masks)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for spatial_inputs, tabular_inputs, flood_masks in val_loader:
                spatial_inputs = spatial_inputs.to(device)
                tabular_inputs = tabular_inputs.to(device)

                flood_masks = flood_masks.to(device)
                if len(flood_masks.shape) == 3:
                    flood_masks = flood_masks.unsqueeze(1)

                predictions = model(spatial_inputs, tabular_inputs)
                val_loss = combined_loss(predictions, flood_masks)
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save epoch metrics.
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                current_lr,
            ])

        # Scheduler is driven by held-out validation loss.
        scheduler.step(avg_val_loss)

        # Keep only the best validation checkpoint.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

            print(
                f"  -> New best validation checkpoint "
                f"(epoch {best_epoch}, val_loss={best_val_loss:.4f})"
            )

    # Record final model-selection information.
    run_config.update({
        "model_selection_metric": "validation_loss",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": save_path.name,
    })

    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=4)

    print("\nTraining complete.")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint     : {save_path}")


if __name__ == "__main__":
    train_model()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader
import sys
import csv
from pathlib import Path

# Configurar rutas de forma dinámica
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Importamos la arquitectura y el dataset
from utils.data_loader import CycloneFloodDataset
from models.multimodal_deeplab import MultimodalDeepLabV3Plus

# 1. Definición de la Boundary Loss (Pérdida de Contorno)
class EdgeBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_edges(self, tensor):
        # Usamos Pooling para simular operaciones morfológicas
        # Requiere tensores 4D: [Batch, Channels, Height, Width]
        dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
        return dilated - eroded

    def forward(self, y_pred_logits, y_true):
        # Convertimos los logits a probabilidades [0, 1]
        y_pred_probs = torch.sigmoid(y_pred_logits)
        
        # Extraemos los contornos
        pred_edges = self.extract_edges(y_pred_probs)
        true_edges = self.extract_edges(y_true)
        
        return F.mse_loss(pred_edges, true_edges)

def train_deeplab():
    # 2. Configuración de Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando DeepLabV3+ Multimodal en: {device}")

    # 3. Inicializar Modelo
    model = MultimodalDeepLabV3Plus(spatial_channels=2, tabular_dim=3, num_classes=1).to(device)

    # --- CONFIGURACIÓN DEL LOGGER CSV ---
    log_path = Path(__file__).resolve().parent / "deeplab_training_logs.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'lr']) # Encabezados
    print(f"Los logs de entrenamiento se guardarán automáticamente en: {log_path}")

    # 4. Dataset y DataLoader (Con Augmentación Física Estricta D4)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

    dataset = CycloneFloodDataset(
        csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), 
        img_dir=str(ROOT_DIR / 'Dataset/'), 
        transform=train_transform
    )
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # 5. La "Pérdida Quimera" (Chimera Loss) COMPLETA
    dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    focal_loss = smp.losses.FocalLoss(smp.losses.BINARY_MODE, alpha=0.99)
    boundary_loss = EdgeBoundaryLoss()
    
    # Peso masivo para obligar a la red a encontrar la clase minoritaria (Agua)
    pos_weight = torch.tensor([50.0]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def combined_loss(y_pred, y_true):
        # 30% Área geométrica
        l_dice = 0.3 * dice_loss(y_pred, y_true)
        # 30% Fuerza bruta para romper la minoría
        l_bce = 0.3 * bce_loss(y_pred, y_true)
        # 20% Castigo a la confianza falsa (Focal Loss)
        l_focal = 0.2 * focal_loss(y_pred, y_true)
        # 20% Precisión topográfica en contornos
        l_bound = 0.2 * boundary_loss(y_pred, y_true)
        
        return l_dice + l_bce + l_focal + l_bound
    
    # 6. Optimizador y Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 150
    print(f"Iniciando entrenamiento de Alta Resolución por {epochs} épocas...")

    # 7. Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for spatial_inputs, tabular_inputs, masks in train_loader:
            spatial_inputs = spatial_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)
            
            # --- PROTECCIÓN DIMENSIONAL ---
            masks = masks.to(device)
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(spatial_inputs, tabular_inputs)
            loss = combined_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # --- GUARDAR EN CSV ---
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, current_lr])

        scheduler.step(avg_loss)

    # 8. GUARDADO SEGURO
    # Cambiamos el nombre para reflejar que es el modelo completo y final
    save_path = str(Path(__file__).resolve().parent / "multimodal_deeplab_v3_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n¡Entrenamiento completado! Pesos guardados intactos en: {save_path}")

if __name__ == "__main__":
    train_deeplab()
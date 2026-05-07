import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Definición de la Boundary Loss aislada
class EdgeBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_edges(self, tensor):
        # max_pool2d requiere entrada 4D: [Batch, Channels, Height, Width]
        dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
        return dilated - eroded

    def forward(self, y_pred_logits, y_true):
        y_pred_probs = torch.sigmoid(y_pred_logits)
        pred_edges = self.extract_edges(y_pred_probs)
        true_edges = self.extract_edges(y_true)
        return F.mse_loss(pred_edges, true_edges)

def run_test_pipeline():
    print("Iniciando Test de Dimensiones de Tensores...\n")
    
    batch_size = 16
    channels = 1
    height = 512
    width = 512

    # 2. Simulamos la salida (logits) del modelo DeepLabV3+
    # Shape: [16, 1, 512, 512]
    mock_logits = torch.randn(batch_size, channels, height, width)
    print(f"[OK] Logits generados con shape: {mock_logits.shape}")

    # 3. Simulamos lo que a veces entrega el DataLoader (Máscaras sin canal)
    # Shape problemático común: [16, 512, 512]
    mock_masks_raw = torch.randint(0, 2, (batch_size, height, width)).float()
    print(f"[ATENCIÓN] Máscaras crudas del DataLoader shape: {mock_masks_raw.shape}")

    # 4. LA SOLUCIÓN DIMENSIONAL
    # Si intentamos pasar mock_masks_raw al EdgeBoundaryLoss, colapsará.
    # Inyectamos la dimensión del canal explícitamente:
    if len(mock_masks_raw.shape) == 3:
        mock_masks_fixed = mock_masks_raw.unsqueeze(1)
        print(f"[CORREGIDO] Máscaras arregladas con unsqueeze(1) shape: {mock_masks_fixed.shape}")
    else:
        mock_masks_fixed = mock_masks_raw

    # 5. Prueba de Fuego de la Función de Pérdida
    try:
        boundary_criterion = EdgeBoundaryLoss()
        loss_value = boundary_criterion(mock_logits, mock_masks_fixed)
        print(f"\n✅ ÉXITO: El EdgeBoundaryLoss procesó los tensores correctamente.")
        print(f"✅ Valor de la pérdida calculada: {loss_value.item():.4f}")
    except Exception as e:
        print(f"\n❌ ERROR FATAL en el procesamiento de tensores:")
        print(e)

if __name__ == "__main__":
    run_test_pipeline()
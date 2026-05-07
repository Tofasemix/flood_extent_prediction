import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_unet import MultimodalFloodModel

def calculate_regional_metrics(pred_mask, true_mask, patch_size=64):
    h, w = true_mask.shape
    
    tp, fp, fn = 0, 0, 0
    
    # Iterate over the image in grids of size patch_size x patch_size
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Extract the regional patch
            pred_patch = pred_mask[i:i+patch_size, j:j+patch_size]
            true_patch = true_mask[i:i+patch_size, j:j+patch_size]
            
            # Is there any real flood in this region?
            has_true_flood = np.sum(true_patch) > 0
            # Did the model predict any flood in this region?
            has_pred_flood = np.sum(pred_patch) > 0
            
            if has_true_flood and has_pred_flood:
                tp += 1
            elif not has_true_flood and has_pred_flood:
                fp += 1
            elif has_true_flood and not has_pred_flood:
                fn += 1
                
    return tp, fp, fn

def run_regional_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Evaluación Regional en: {device}")

    # Load Model
    model = MultimodalFloodModel(tabular_dim=3, bottleneck_dim=512).to(device)
    weights_path = Path(__file__).resolve().parent / "multimodal_flood_unet.pth"
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.eval()

    # Load Dataset
    dataset = CycloneFloodDataset(csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), img_dir=str(ROOT_DIR / 'Dataset/'), transform=None)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    global_tp = 0
    global_fp = 0
    global_fn = 0
    valid_images = 0

    with torch.no_grad():
        for spatial_inputs, tabular_inputs, ground_truth in tqdm(val_loader, desc="Procesando Ciclones"):
            spatial_inputs = spatial_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)
            
            raw_logits = model(spatial_inputs, tabular_inputs)
            probabilities = torch.sigmoid(raw_logits)
            
            predicted_mask = (probabilities > 0.15).float()

            pred_np = predicted_mask.squeeze().cpu().numpy()
            true_np = ground_truth.squeeze().cpu().numpy()

            # Solo evaluamos imágenes que tienen inundación real
            if np.sum(true_np) > 0:
                # Usamos parches de 64x64 (divide el mapa en 64 sub-regiones)
                tp, fp, fn = calculate_regional_metrics(pred_np, true_np, patch_size=64)
                
                global_tp += tp
                global_fp += fp
                global_fn += fn
                valid_images += 1

    # Cálculos Finales Globales
    epsilon = 1e-8
    regional_iou = global_tp / (global_tp + global_fp + global_fn + epsilon)
    regional_f1 = (2 * global_tp) / ((2 * global_tp) + global_fp + global_fn + epsilon)
    regional_precision = global_tp / (global_tp + global_fp + epsilon)
    regional_recall = global_tp / (global_tp + global_fn + epsilon)

    print("\n" + "="*50)
    print("🏆 MÉTRICAS DE EVALUACIÓN REGIONAL (Parches 64x64) 🏆")
    print("="*50)
    print(f"Ciclones Evaluados             : {valid_images}")
    print(f"IoU Regional (Intersection over Union): {regional_iou:.4f}")
    print(f"F1-Score Regional (Dice)              : {regional_f1:.4f}")
    print(f"Precisión (¿Cuántas alertas fueron reales?): {regional_precision:.4f}")
    print(f"Recall (Sensibilidad a regiones reales)    : {regional_recall:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_regional_evaluation()
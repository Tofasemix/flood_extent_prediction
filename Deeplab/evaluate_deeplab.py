import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_deeplab import MultimodalDeepLabV3Plus

def calculate_regional_metrics(pred_mask, true_mask, patch_size=64):
    """Evalúa la precisión regional en parches de 64x64 píxeles."""
    h, w = true_mask.shape
    tp, fp, fn = 0, 0, 0
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            pred_patch = pred_mask[i:i+patch_size, j:j+patch_size]
            true_patch = true_mask[i:i+patch_size, j:j+patch_size]
            
            has_true_flood = np.sum(true_patch) > 0
            has_pred_flood = np.sum(pred_patch) > 0
            
            if has_true_flood and has_pred_flood:
                tp += 1
            elif not has_true_flood and has_pred_flood:
                fp += 1
            elif has_true_flood and not has_pred_flood:
                fn += 1
                
    return tp, fp, fn

def run_deeplab_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Evaluación Regional de DeepLabV3+ en: {device}")

    # 1. Cargar la nueva arquitectura
    model = MultimodalDeepLabV3Plus(spatial_channels=2, tabular_dim=3, num_classes=1).to(device)
    
    # 2. Cargar los pesos que acabas de entrenar
    weights_path = Path(__file__).resolve().parent / "multimodal_deeplab_v3.pth"
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.eval() # Modo evaluación estricto

    # 3. Dataset puro (Sin augmentación, queremos evaluar sobre la física real)
    dataset = CycloneFloodDataset(csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), img_dir=str(ROOT_DIR / 'Dataset/'), transform=None)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    global_tp, global_fp, global_fn = 0, 0, 0
    valid_images = 0

    with torch.no_grad():
        for spatial_inputs, tabular_inputs, ground_truth in tqdm(val_loader, desc="Evaluando Ciclones"):
            spatial_inputs = spatial_inputs.to(device)
            tabular_inputs = tabular_inputs.to(device)
            
            raw_logits = model(spatial_inputs, tabular_inputs)
            probabilities = torch.sigmoid(raw_logits)
            
            # Usamos el mismo umbral "valiente" del 15% para una comparación justa
            predicted_mask = (probabilities > 0.15).float()

            pred_np = predicted_mask.squeeze().cpu().numpy()
            true_np = ground_truth.squeeze().cpu().numpy()

            if np.sum(true_np) > 0:
                tp, fp, fn = calculate_regional_metrics(pred_np, true_np, patch_size=64)
                global_tp += tp
                global_fp += fp
                global_fn += fn
                valid_images += 1

    # 4. Matemáticas Finales
    epsilon = 1e-8
    regional_iou = global_tp / (global_tp + global_fp + global_fn + epsilon)
    regional_f1 = (2 * global_tp) / ((2 * global_tp) + global_fp + global_fn + epsilon)
    regional_precision = global_tp / (global_tp + global_fp + epsilon)
    regional_recall = global_tp / (global_tp + global_fn + epsilon)

    print("\n" + "="*50)
    print("🏆 MÉTRICAS DE DEEPLABv3+ (Parches 64x64) 🏆")
    print("="*50)
    print(f"Ciclones Evaluados             : {valid_images}")
    print(f"IoU Regional                   : {regional_iou:.4f}")
    print(f"F1-Score Regional              : {regional_f1:.4f} ")
    print(f"Precisión                      : {regional_precision:.4f}")
    print(f"Recall                         : {regional_recall:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_deeplab_evaluation()
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset

# Importamos la nueva arquitectura
from models.multimodal_deeplab import MultimodalDeepLabV3Plus

def run_deeplab_inference():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Inferencia Visual en: {device}")

    # 2. Inicializar Modelo y Cargar Pesos
    model = MultimodalDeepLabV3Plus(spatial_channels=2, tabular_dim=3, num_classes=1).to(device)
    weights_path = Path(__file__).resolve().parent / "multimodal_deeplab_v3.pth"
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.eval() # Modo evaluación para congelar BatchNorm

    # 3. Cargar el Dataset (Sin Augmentación para ver la física real)
    dataset = CycloneFloodDataset(csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), img_dir=str(ROOT_DIR / 'Dataset/'), transform=None)
    
    # Seleccionar un ciclón aleatorio (o puedes fijar sample_idx a un número específico para tu video)
    sample_idx = random.randint(0, len(dataset) - 1)
    spatial_tensor, tabular_tensor, ground_truth_mask = dataset[sample_idx]

    # Añadir dimensión de batch y mover a GPU
    spatial_input = spatial_tensor.unsqueeze(0).to(device)
    tabular_input = tabular_tensor.unsqueeze(0).to(device)

    # 4. Forward Pass
    with torch.no_grad(): 
        raw_logits = model(spatial_input, tabular_input)
        probabilities = torch.sigmoid(raw_logits)
        
        print("\n--- DIAGNÓSTICO DE PREDICCIÓN ---")
        print(f"ID del Ciclón Evaluado : {sample_idx}")
        print(f"Logit máximo           : {raw_logits.max().item():.4f}")
        print(f"Probabilidad máxima    : {probabilities.max().item():.6f}")
        print(f"Probabilidad media     : {probabilities.mean().item():.6f}")
        
        # Usamos el umbral del 15% que validamos en nuestras métricas
        predicted_mask = (probabilities > 0.15).float()

    # 5. Extracción de Datos para Graficar
    tp36_img = spatial_tensor[0].cpu().numpy()      # Lluvia
    slope_img = spatial_tensor[1].cpu().numpy()     # Topografía
    true_mask = ground_truth_mask[0].cpu().numpy()  # Ground Truth
    prob_map = probabilities.squeeze().cpu().numpy()# Mapa de Calor de Probabilidades
    pred_mask = predicted_mask.squeeze().cpu().numpy() # Máscara Binaria Final

    # 6. Generar el Gráfico Académico (5 Columnas)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    plt.suptitle(f"DeepLabV3+ Multimodal - Resultados de Inferencia (Muestra ID: {sample_idx})", fontsize=18, fontweight='bold')

    # Topografía
    axes[0].imshow(slope_img, cmap='terrain')
    axes[0].set_title('1. Topografía (SLOPE)')
    axes[0].axis('off')

    # Precipitación
    axes[1].imshow(tp36_img, cmap='Blues')
    axes[1].set_title('2. Precipitación (TP36)')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(true_mask, cmap='gray')
    axes[2].set_title('3. Inundación Real (Ground Truth)')
    axes[2].axis('off')

    # Mapa de Probabilidades (El "cerebro" del modelo)
    im = axes[3].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('4. Mapa de Probabilidad (0.0 a 1.0)')
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    # Predicción Binaria
    axes[4].imshow(pred_mask, cmap='magma')
    axes[4].set_title('5. Predicción Final (Umbral > 15%)')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_deeplab_inference()
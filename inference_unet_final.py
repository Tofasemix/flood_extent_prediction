import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import sys

# Resolución dinámica
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_unet import MultimodalFloodModel

def run_unet_inference():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Inferencia Visual de U-Net en: {device}")

    # 2. Inicializar Modelo y Cargar Pesos de la U-Net ganadora
    model = MultimodalFloodModel(tabular_dim=3, bottleneck_dim=512).to(device)
    
    weights_path = ROOT_DIR / "MyUnet" / "multimodal_flood_unet.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Error: No se encontraron los pesos en {weights_path}")
        return
        
    model.eval()

    # 3. Cargar el Dataset (Sin Augmentación)
    dataset = CycloneFloodDataset(csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), 
                                  img_dir=str(ROOT_DIR / 'Dataset/'), transform=None)
    
    # Seleccionar un ciclón aleatorio (o uno fijo si ya tienes uno favorito)
    sample_idx = random.randint(0, len(dataset) - 1)
    spatial_tensor, tabular_tensor, ground_truth_mask = dataset[sample_idx]

    spatial_input = spatial_tensor.unsqueeze(0).to(device)
    tabular_input = tabular_tensor.unsqueeze(0).to(device)

    # 4. Forward Pass
    with torch.no_grad(): 
        raw_logits = model(spatial_input, tabular_input)
        probabilities = torch.sigmoid(raw_logits)
        
        print("\n--- DIAGNÓSTICO DE PREDICCIÓN (U-NET) ---")
        print(f"ID del Ciclón Evaluado : {sample_idx}")
        print(f"Probabilidad máxima    : {probabilities.max().item():.6f}")
        
        # Umbral del 15% (0.15) o 50% (0.5), puedes jugar con esto según la confianza de la U-Net
        predicted_mask = (probabilities > 0.15).float()

    # 5. Extracción para Gráficas
    tp36_img = spatial_tensor[0].cpu().numpy()      
    slope_img = spatial_tensor[1].cpu().numpy()     
    true_mask = ground_truth_mask[0].cpu().numpy()  
    prob_map = probabilities.squeeze().cpu().numpy()
    pred_mask = predicted_mask.squeeze().cpu().numpy()

    # 6. Gráfico Académico (5 Columnas)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    plt.suptitle(f"U-Net Multimodal - Resultados de Inferencia (Muestra ID: {sample_idx})", fontsize=18, fontweight='bold')

    axes[0].imshow(slope_img, cmap='terrain')
    axes[0].set_title('1. Topografía (SLOPE)')
    axes[0].axis('off')

    axes[1].imshow(tp36_img, cmap='Blues')
    axes[1].set_title('2. Precipitación (TP36)')
    axes[1].axis('off')

    axes[2].imshow(true_mask, cmap='gray')
    axes[2].set_title('3. Inundación Real (Ground Truth)')
    axes[2].axis('off')

    im = axes[3].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('4. Mapa de Probabilidad (0.0 a 1.0)')
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(pred_mask, cmap='magma')
    axes[4].set_title('5. Predicción Final (Umbral > 15%)')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_unet_inference()
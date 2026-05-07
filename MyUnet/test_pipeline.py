import torch
from torch.utils.data import DataLoader
import albumentations as A
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.data_loader import CycloneFloodDataset
from models.multimodal_unet import MultimodalFloodModel

def test_pipeline():
    print("Iniciando prueba del Pipeline Multimodal...\n")
    
    # 1. Probar el Dataloader y las Transformaciones
    print("1️⃣ Probando CycloneFloodDataset...")
    try:
        # Usamos una transformación de prueba básica
        test_transform = A.Compose([A.HorizontalFlip(p=1.0)])
        
        # Usamos merged_tabular_data.csv (los 167 ciclones). 
        # Recuerda que nuestra clase ya ignora el 'radius' internamente.
        dataset = CycloneFloodDataset(
            csv_file=str(ROOT_DIR / 'clean_tabular_data.csv'), 
            img_dir=str(ROOT_DIR / 'Dataset/'), 
            transform=test_transform
        )
        
        # Cargamos un batch de 4 ciclones
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        spatial_batch, tabular_batch, mask_batch = next(iter(loader))
        
        print(f"✅ Dataloader exitoso.")
        print(f"   -> Spatial Tensor: {spatial_batch.shape} (Esperado: [4, 2, 512, 512])")
        print(f"   -> Tabular Tensor: {tabular_batch.shape} (Esperado: [4, 3])")
        print(f"   -> Target Mask:    {mask_batch.shape} (Esperado: [4, 1, 512, 512])\n")
        
    except Exception as e:
        print(f"❌ Error en el Dataloader: {e}")
        return

    # 2. Probar la Arquitectura Híbrida U-Net
    print("2️⃣ Probando MultimodalFloodModel...")
    try:
        # Si tienes la GPU disponible, movemos el modelo ahí; si no, en CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   -> Usando dispositivo: {device}")
        
        model = MultimodalFloodModel(tabular_dim=3, bottleneck_dim=512).to(device)
        
        # Movemos los tensores al mismo dispositivo
        spatial_batch = spatial_batch.to(device)
        tabular_batch = tabular_batch.to(device)
        
        print(f"✅ Modelo instanciado correctamente en {device}.\n")
        
    except Exception as e:
        print(f"❌ Error al instanciar el modelo: {e}")
        return

    # 3. Probar el Forward Pass (El cuello de botella de fusión)
    print("3️⃣ Probando Forward Pass (Fusión de Datos)...")
    try:
        # Pasamos las imágenes y los datos tabulares por la red
        predictions = model(spatial_batch, tabular_batch)
        
        print(f"✅ Forward Pass exitoso.")
        print(f"   -> Output Tensor: {predictions.shape} (Esperado: [4, 1, 512, 512])\n")
        print("🚀 ¡EL PIPELINE ESTÁ LISTO PARA ENTRENAR! 🚀")
        
    except Exception as e:
        print(f"❌ Error durante el Forward Pass: {e}")

if __name__ == "__main__":
    test_pipeline()
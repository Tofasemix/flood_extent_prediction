import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparative_loss():
    # Rutas dinámicas a ambos archivos
    root_dir = Path(__file__).resolve().parent
    unet_csv = root_dir / "MyUnet" / "unet_training_logs.csv"
    deeplab_csv = root_dir / "Deeplab" / "deeplab_training_logs.csv"
    
    print("Cargando logs de U-Net y DeepLabV3+...")

    try:
        df_unet = pd.read_csv(unet_csv)
        df_deeplab = pd.read_csv(deeplab_csv)
    except FileNotFoundError as e:
        print(f"❌ Error al cargar los archivos: {e}")
        return

    # Generar la gráfica comparativa
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Curva U-Net (Morado)
    ax.plot(df_unet['epoch'], df_unet['loss'], color='tab:purple', 
            linewidth=2, label='U-Net Baseline (BCE + Dice)')
    
    # Curva DeepLabV3+ (Rojo)
    ax.plot(df_deeplab['epoch'], df_deeplab['loss'], color='tab:red', 
            linewidth=2.5, label='DeepLabV3+ (Chimera Loss)')

    # Configuración de Ejes y Textos
    ax.set_xlabel('Época (Epoch)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Comparativa de Convergencia: U-Net vs DeepLabV3+', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Ajuste dinámico del eje Y
    all_losses = pd.concat([df_unet['loss'], df_deeplab['loss']])
    ax.set_ylim(bottom=max(0, all_losses.min() - 0.05), top=all_losses.max() + 0.05)

    # Leyenda y malla
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Guardar y mostrar
    output_filename = 'comparative_training_curve.png'
    fig.tight_layout()  
    plt.savefig(output_filename)
    print(f"✅ Gráfica comparativa generada exitosamente: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_comparative_loss()
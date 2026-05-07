import os
import pandas as pd

# Carga el CSV limpio
df = pd.read_csv('clean_tabular_data.csv')

# Directorio
base_dir = 'Dataset' 

# Diccionario que mapea la 'Carpeta' -> 'Prefijo del archivo'
folders_and_prefixes = {
    'FLOOD': 'FLOOD',
    'PRECIPITATION': 'TP36',  # Carpeta PRECIPITATION, archivo TP36_...
    'SLOPE': 'SLOPE'
}

missing_files = []

print(f"Auditing {len(df)} cyclones...")

for index, row in df.iterrows():
    cyclone_id = row['name_date']
    
    for folder, prefix in folders_and_prefixes.items():
        # Construye el nombre exacto usando el prefijo correcto
        file_name = f"{prefix}_{cyclone_id}.png"
        file_path = os.path.join(base_dir, folder, file_name)
        
        # Revisa si existe
        if not os.path.exists(file_path):
            missing_files.append(file_path)

if not missing_files:
    print(f"✅ ¡Éxito! Los {len(df)} ciclones tienen sus 3 imágenes enlazadas perfectamente.")
else:
    print(f"❌ Se encontraron {len(missing_files)} enlaces rotos. Aquí están los primeros:")
    for f in missing_files[:5]:
        print(f)
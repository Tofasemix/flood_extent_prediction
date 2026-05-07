import pandas as pd

# Cargar el dataset combinado que generamos
df = pd.read_csv('merged_tabular_data.csv')

# Eliminar la columna 'radius'
clean_df = df.drop('radius', axis = 1)

# Guardar el nuevo dataset limpio
clean_df.to_csv('clean_tabular_data.csv', index=False)

print(f"Tuplas originales: {len(df)}")
print(f"Tuplas limpias listas para PyTorch: {len(clean_df)}")
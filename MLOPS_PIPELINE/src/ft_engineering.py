# mlops_pipeline/src/ft_engineering.py

import pandas as pd
import numpy as np
import joblib
import cloudpickle
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

# Importar módulo de carga
from cargar_datos import cargarDatos

print("="*60)
print("🔧 INGENIERÍA DE CARACTERÍSTICAS")
print("="*60)


# ============================================
# FUNCIÓN AUXILIAR PARA CONVERTIR A STRING
# ============================================

def convertir_a_string(X):
    """Convierte todas las columnas a tipo string"""
    return X.astype(str)


# ============================================
# 1. CARGAR DATOS
# ============================================

df = cargarDatos()
print(f"✅ Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")


# ============================================
# 2. VARIABLES SEGURAS (SOLO PRE-PRÉSTAMO)
# ============================================

target = 'Pago_atiempo'

# Variables numéricas seguras (conocidas antes de otorgar el préstamo)
variables_numericas = [
    'capital_prestado',             # Monto del préstamo
    'plazo_meses',                  # Plazo del préstamo
    'edad_cliente',                 # Edad del cliente
    'salario_cliente',              # Ingresos del cliente
    'total_otros_prestamos',        # Nivel de endeudamiento
    'cuota_pactada',                # Cuota a pagar
    'creditos_sectorFinanciero',    # Créditos en sector financiero
    'creditos_sectorCooperativo',   # Créditos en cooperativas
    'creditos_sectorReal',          # Créditos en sector real
    'promedio_ingresos_datacredito', # Ingresos promedio
]

# Variables categóricas (todas eliminadas por baja correlación)
variables_categoricas = []

# Unir todas las variables a conservar
variables_conservar = variables_numericas + variables_categoricas

# Filtrar solo las que existen en el DataFrame
variables_conservar = [col for col in variables_conservar if col in df.columns]

print(f"\n📌 Variables seleccionadas: {len(variables_conservar)}")
print(f"   {variables_conservar}")

# ============================================
# 3. CREAR DATAFRAME CON VARIABLES SELECCIONADAS
# ============================================

variables_finales = variables_conservar + [target]
df_seleccionado = df[variables_finales].copy()

print(f"\n✅ DataFrame seleccionado: {df_seleccionado.shape[0]} filas x {df_seleccionado.shape[1]} columnas")

# ============================================
# 4. SEPARAR X E Y
# ============================================

X = df_seleccionado.drop(target, axis=1)
y = df_seleccionado[target]

print(f"\n🎯 Variable objetivo: '{target}'")
print(f"   Distribución:\n{y.value_counts()}")

# ============================================
# 5. IDENTIFICAR TIPOS DE VARIABLES
# ============================================

num_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\n📌 Variables numéricas: {len(num_features)}")
print(f"📌 Variables categóricas: {len(cat_features)}")

# ============================================
# 6. CREAR PIPELINES DE TRANSFORMACIÓN
# ============================================

# Ruta numérica
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Ruta categórica
cat_transformer = Pipeline([
    ('to_str', FunctionTransformer(convertir_a_string)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("\n✅ Pipeline de transformación creado")

# ============================================
# 7. DIVIDIR DATOS EN TRAIN/TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 División:")
print(f"  - Train: {len(X_train)}")
print(f"  - Test: {len(X_test)}")

# ============================================
# 8. APLICAR PREPROCESAMIENTO
# ============================================

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(f"\n✅ Datos transformados:")
print(f"  - X_train: {X_train_transformed.shape}")
print(f"  - X_test: {X_test_transformed.shape}")

# ============================================
# 9. GUARDAR PREPROCESADOR (con cloudpickle)
# ============================================

os.makedirs("../models", exist_ok=True)

# Guardar usando cloudpickle (permite serializar funciones correctamente)
with open("../models/preprocessor.pkl", "wb") as f:
    cloudpickle.dump(preprocessor, f)
print("✅ preprocessor.pkl guardado (con cloudpickle)")

# ============================================
# 10. GUARDAR DATOS TRANSFORMADOS
# ============================================

with open("../models/datos_preprocesados.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train_transformed,
        'X_test': X_test_transformed,
        'y_train': y_train,
        'y_test': y_test
    }, f)
print("✅ datos_preprocesados.pkl guardado")

print("\n" + "="*60)
print("✅ INGENIERÍA DE CARACTERÍSTICAS COMPLETADA")
print("="*60)
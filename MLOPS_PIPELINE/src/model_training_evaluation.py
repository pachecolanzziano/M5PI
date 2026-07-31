import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 ENTRENAMIENTO Y EVALUACION DE MODELOS")
print("🎯 OBJETIVO: MAXIMIZAR RECALL DE CLASE 0 (No Paga)")
print("="*60)

# ============================================
# 1. CARGAR DATOS PREPROCESADOS
# ============================================

print("\n📂 Cargando datos preprocesados...")

try:
    with open("../models/datos_preprocesados.pkl", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
    print("✅ Datos cargados desde models/datos_preprocesados.pkl")
except FileNotFoundError:
    print("❌ Error: No se encuentra datos_preprocesados.pkl")
    print("   Ejecuta primero: python ft_engineering.py")
    exit()

# ============================================
# 2. CONVERTIR A ARRAYS MODIFICABLES
# ============================================

y_train = np.array(y_train).copy()
y_test = np.array(y_test).copy()

if not isinstance(X_train, np.ndarray):
    X_train = np.array(X_train)
    X_test = np.array(X_test)

print(f"✅ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")

# ============================================
# 3. DEFINIR MODELOS
# ============================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

print("\n📋 Modelos a evaluar:")
for name in models.keys():
    print(f"  - {name}")

# ============================================
# 4. FUNCION PARA EVALUAR RECALL DE CLASE 0
# ============================================

def calcular_recall_clase0(model, X_test, y_test):
    """Calcula el Recall de la clase 0 (No Paga)"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    recall_clase0 = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    return recall_clase0, cm

# ============================================
# 5. ENTRENAR Y EVALUAR CADA MODELO
# ============================================

print("\n" + "="*60)
print("📊 Entrenando y evaluando modelos...")
print("="*60)

mejor_recall = -1
mejor_modelo = None
mejor_nombre = None

for name, model in models.items():
    print(f"\n▶ Entrenando: {name}")
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Calcular Recall de Clase 0
    recall_clase0, cm = calcular_recall_clase0(model, X_test, y_test)
    
    print(f"   Recall Clase 0 (No Paga): {recall_clase0:.4f}")
    print(f"   VP: {cm[0][0]} | FP: {cm[0][1]} | FN: {cm[1][0]} | VN: {cm[1][1]}")
    
    # Guardar el que tenga mejor Recall de Clase 0
    if recall_clase0 > mejor_recall:
        mejor_recall = recall_clase0
        mejor_modelo = model
        mejor_nombre = name

# ============================================
# 6. GUARDAR EL MEJOR MODELO
# ============================================

print("\n" + "="*60)
print("🏆 MEJOR MODELO SELECCIONADO")
print("="*60)

print(f"\n✅ Modelo: {mejor_nombre}")
print(f"   Recall Clase 0 (No Paga): {mejor_recall:.4f}")

# Guardar el mejor modelo
os.makedirs("../models", exist_ok=True)
joblib.dump(mejor_modelo, "../models/best_model.pkl")
print("✅ Mejor modelo guardado en models/best_model.pkl")

print("\n" + "="*60)
print("✅ MODELADO COMPLETADO")
print("="*60)
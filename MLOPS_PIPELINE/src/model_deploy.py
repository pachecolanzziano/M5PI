# src/model_deploy.py

import pandas as pd
import numpy as np
import joblib
import cloudpickle
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ============================================
# 1. INICIALIZACIÓN DE LA APLICACIÓN
# ============================================

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="Esta API permite predecir si un cliente pagará a tiempo o no, basado en información histórica de créditos.",
    version="1.1.1"
)

# ============================================
# 2. FUNCIÓN AUXILIAR (necesaria para cargar el preprocesador)
# ============================================

def convertir_a_string(X):
    """Convierte todas las columnas a tipo string"""
    return X.astype(str)


# ============================================
# 3. CARGAR MODELO Y PREPROCESADOR CON RUTAS DINÁMICAS
# ============================================

# Obtener la ruta del directorio donde está model_deploy.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construir rutas dinámicamente (sube un nivel a la raíz del proyecto)
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "best_model.pkl"))
PREPROCESSOR_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl"))

print(f"📂 Buscando modelo en: {MODEL_PATH}")
print(f"📂 Buscando preprocesador en: {PREPROCESSOR_PATH}")

# Variables globales
modelo = None
preprocessor = None

# ============================================
# 3.1 CARGAR MODELO (con joblib)
# ============================================

try:
    modelo = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado exitosamente")
except FileNotFoundError:
    print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
    modelo = None
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None

# ============================================
# 3.2 CARGAR PREPROCESADOR (con cloudpickle)
# ============================================

try:
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = cloudpickle.load(f)
    print("✅ Preprocesador cargado exitosamente")
except FileNotFoundError:
    print(f"❌ Error: No se encuentra el preprocesador en {PREPROCESSOR_PATH}")
    preprocessor = None
except Exception as e:
    print(f"❌ Error al cargar el preprocesador: {e}")
    preprocessor = None

# ============================================
# 4. ESQUEMAS DE DATOS (Pydantic)
# ============================================

class ClienteInput(BaseModel):
    """Esquema para la entrada de datos de un cliente"""
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "capital_prestado": 3000000.0,
                "plazo_meses": 12,
                "edad_cliente": 35,
                "salario_cliente": 2000000.0,
                "total_otros_prestamos": 500000.0,
                "cuota_pactada": 250000.0,
                "creditos_sectorFinanciero": 2,
                "creditos_sectorCooperativo": 1,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 1800000.0
            }
        }

class ClienteInputBatch(BaseModel):
    """Esquema para entrada de múltiples clientes"""
    clientes: List[ClienteInput]

class PrediccionResponse(BaseModel):
    """Esquema para la respuesta de predicción individual"""
    prediccion: int
    probabilidad_pago: float
    probabilidad_no_pago: float
    mensaje: str

class PrediccionBatchResponse(BaseModel):
    """Esquema para la respuesta de predicción múltiple"""
    predicciones: List[PrediccionResponse]

# ============================================
# 5. FUNCIÓN DE PREPROCESAMIENTO
# ============================================

def preparar_datos_para_prediccion(datos: dict, preprocessor):
    """
    Prepara los datos para la predicción usando el preprocesador.
    """
    # Crear DataFrame con los datos
    df = pd.DataFrame([datos])
    
    # Obtener las variables que usa el preprocesador
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    features_used = list(num_features)
    
    # Filtrar columnas que existen
    features_available = [col for col in features_used if col in df.columns]
    X_prepared = df[features_available]
    
    # Transformar
    X_transformed = preprocessor.transform(X_prepared)
    
    return X_transformed

# ============================================
# 6. ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Endpoint raíz con mensaje de bienvenida"""
    return {
        "mensaje": "Bienvenido a la API de Predicción de Pago a Tiempo",
        "documentacion": "/docs",
        "documentacion_redoc": "/redoc",
        "estado_modelo": "Cargado" if modelo is not None else "No cargado",
        "estado_preprocesador": "Cargado" if preprocessor is not None else "No cargado"
    }

@app.get("/saludo")
def saludo():
    """Endpoint de saludo"""
    return {"mensaje": "Hola! Esta API está funcionando correctamente y usa un modelo de Machine Learning."}

@app.get("/health")
def health_check():
    """Verifica el estado de la API y del modelo"""
    if modelo is None or preprocessor is None:
        return {
            "status": "ERROR",
            "modelo_cargado": modelo is not None,
            "preprocesador_cargado": preprocessor is not None,
            "mensaje": "El modelo o el preprocesador no están cargados correctamente."
        }
    return {
        "status": "OK",
        "modelo_cargado": True,
        "preprocesador_cargado": True,
        "mensaje": "La API está funcionando correctamente."
    }

@app.post("/predict", response_model=PrediccionResponse)
def predict(cliente: ClienteInput):
    """
    Realiza una predicción para un cliente individual.
    """
    # Verificar que el modelo y preprocesador estén cargados
    if modelo is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo o el preprocesador no están disponibles. Contacte al administrador."
        )
    
    try:
        # Preparar datos
        datos_dict = cliente.dict()
        X_transformed = preparar_datos_para_prediccion(datos_dict, preprocessor)
        
        # Hacer predicción
        y_pred = modelo.predict(X_transformed)[0]
        y_prob = modelo.predict_proba(X_transformed)[0][1]
        
        # Determinar mensaje
        if y_pred == 1:
            mensaje = "El cliente probablemente PAGA a tiempo"
        else:
            mensaje = "El cliente probablemente NO PAGA a tiempo"
        
        return PrediccionResponse(
            prediccion=int(y_pred),
            probabilidad_pago=float(y_prob),
            probabilidad_no_pago=float(1 - y_prob),
            mensaje=mensaje
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al hacer la predicción: {str(e)}"
        )

@app.post("/predict_batch", response_model=PrediccionBatchResponse)
def predict_batch(clientes: ClienteInputBatch):
    """
    Realiza predicciones para múltiples clientes.
    """
    # Verificar que el modelo y preprocesador estén cargados
    if modelo is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo o el preprocesador no están disponibles. Contacte al administrador."
        )
    
    try:
        resultados = []
        
        for cliente in clientes.clientes:
            datos_dict = cliente.dict()
            X_transformed = preparar_datos_para_prediccion(datos_dict, preprocessor)
            
            y_pred = modelo.predict(X_transformed)[0]
            y_prob = modelo.predict_proba(X_transformed)[0][1]
            
            if y_pred == 1:
                mensaje = "El cliente probablemente PAGA a tiempo"
            else:
                mensaje = "El cliente probablemente NO PAGA a tiempo"
            
            resultados.append(PrediccionResponse(
                prediccion=int(y_pred),
                probabilidad_pago=float(y_prob),
                probabilidad_no_pago=float(1 - y_prob),
                mensaje=mensaje
            ))
        
        return PrediccionBatchResponse(predicciones=resultados)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al hacer las predicciones: {str(e)}"
        )

# ============================================
# 7. EJECUCIÓN LOCAL
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
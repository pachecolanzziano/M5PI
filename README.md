# M5PI
Proyecto integrador modulo 5 


# 📊 Proyecto MLOps - Predicción de Comportamiento Crediticio

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo **desarrollar un modelo predictivo** que anticipe el comportamiento de pago de nuevos usuarios en una empresa financiera. Utilizando información histórica de créditos, el modelo clasifica si un cliente **pagará a tiempo (1)** o **no pagará a tiempo (0)**.

El proyecto sigue una arquitectura de MLOps con:
- **Versionamiento** de código y datos
- **Pipeline** de transformación de datos
- **Entrenamiento y evaluación** de modelos supervisados
- **Monitoreo** de Data Drift en producción
- **API** para disponibilizar el modelo
- **Contenedor Docker** para despliegue

---

## 📁 Estructura del Proyecto
mlops_pipeline/
├── src/
│ ├── cargar_datos.py # Carga de datos desde Excel
│ ├── ft_engineering.py # Feature Engineering y preprocesamiento
│ ├── model_training_evaluation.py # Entrenamiento y selección del mejor modelo
│ ├── model_deploy.py # API con FastAPI (Avance #4)
│ └── model_monitoring.py # Dashboard de monitoreo con Streamlit
├── models/
│ ├── best_model.pkl # Mejor modelo entrenado
│ ├── preprocessor.pkl # Preprocesador para transformar datos
│ └── datos_preprocesados.pkl # Datos transformados
├── Base_de_datos.xlsx # Datos históricos
├── Base_de_datos_con_Data_Drift_Simulado.xlsx # Datos nuevos con drift
├── requirements.txt # Dependencias del proyecto
├── .gitignore
├── README.md
└── Dockerfile


---

## 🎯 Caso de Negocio

### Problema

La empresa financiera otorga créditos a sus clientes. Actualmente, **no cuenta con un sistema automatizado** para evaluar el riesgo de impago de nuevos solicitantes. Esto genera:

- **Pérdidas económicas** por créditos otorgados a clientes que no pagan
- **Falta de criterio objetivo** en la aprobación de créditos
- **Ineficiencia operativa** al evaluar manualmente cada solicitud

### Solución Propuesta

Un **modelo de Machine Learning** que, basado en información histórica de créditos, prediga si un nuevo cliente pagará o no a tiempo.

### Beneficios Esperados

| Beneficio | Impacto |
|-----------|---------|
| **Reducción de pérdidas** | Identificar clientes de alto riesgo antes de otorgar el crédito |
| **Decisiones objetivas** | Evaluación basada en datos, no en criterios subjetivos |
| **Eficiencia operativa** | Automatización del proceso de aprobación |
| **Escalabilidad** | El modelo puede manejar grandes volúmenes de solicitudes |

---

## 📊 Datos Utilizados

### Descripción del Dataset

El dataset contiene **10,763 registros** y **23 columnas** con información de créditos históricos.

### Variables Seleccionadas para el Modelo

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `capital_prestado` | Monto del préstamo solicitado | Numérica |
| `plazo_meses` | Plazo del préstamo en meses | Numérica |
| `edad_cliente` | Edad del cliente | Numérica |
| `salario_cliente` | Ingresos del cliente | Numérica |
| `total_otros_prestamos` | Total de otras deudas | Numérica |
| `cuota_pactada` | Cuota mensual pactada | Numérica |
| `creditos_sectorFinanciero` | Créditos en sector financiero | Numérica |
| `creditos_sectorCooperativo` | Créditos en cooperativas | Numérica |
| `creditos_sectorReal` | Créditos en sector real | Numérica |
| `promedio_ingresos_datacredito` | Ingresos promedio según DataCrédito | Numérica |

### Variables Eliminadas

| Variable | Motivo |
|----------|--------|
| `puntaje` | Posible Data Leakage (correlación 0.923) |
| `saldo_mora` | Post-préstamo (Data Leakage) |
| `saldo_total` | Post-préstamo (Data Leakage) |
| `saldo_principal` | Post-préstamo (Data Leakage) |
| `huella_consulta` | Post-préstamo (Data Leakage) |
| `tipo_credito` | Baja correlación con target |
| `tipo_laboral` | Baja correlación con target |
| `tendencia_ingresos` | Baja correlación con target |
| `fecha_prestamo` | Requiere procesamiento especial |
| `puntaje_datacredito` | Redundante con `puntaje` |
| `saldo_mora_codeudor` | Alto % de nulos |

---

## 🏆 Modelo Seleccionado

### Proceso de Selección

Se evaluaron **4 modelos supervisados**:

| Modelo | Recall Clase 0 (No Paga) | F1-Score |
|--------|--------------------------|----------|
| Logistic Regression | 0.0000 | 0.9757 |
| **Random Forest** | 0.0098 | 0.9760 |
| **Gradient Boosting** | **0.0196** | 0.9752 |
| SVM | 0.0000 | 0.9757 |

### Modelo Elegido: **Gradient Boosting**

**Criterio de selección:** Mejor **Recall de Clase 0 (No Paga)**.

### Métricas del Modelo (con umbral 0.5)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 95.17% |
| **Recall No Paga** | 1.96% |
| **Recall Paga** | 99.80% |
| **F1-Score** | 97.52% |
| **ROC-AUC** | 0.6320 |

### Matriz de Confusión
┌─────────────┬─────────────┐
│ VP: 2 │ FP: 100 │
├─────────────┼─────────────┤
│ FN: 4 │ VN: 2047 │
└─────────────┴─────────────┘


| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **VP (Verdaderos Positivos)** | 2 | De 102 No Pagadores, detecta 2 |
| **FP (Falsos Positivos)** | 100 | 100 Pagadores son clasificados como No Pagadores |
| **FN (Falsos Negativos)** | 4 | 4 No Pagadores no son detectados |
| **VN (Verdaderos Negativos)** | 2047 | 2047 Pagadores detectados correctamente |

---

## 🔍 Principales Hallazgos

### 1. Data Leakage Detectado

La variable `puntaje` presentó una **correlación de 0.923** con el target, lo que indica que probablemente contiene información posterior al préstamo. Fue eliminada para evitar sesgos en el modelo.

### 2. Datos Extremadamente Desbalanceados

| Clase | Cantidad | Proporción |
|-------|----------|------------|
| **Paga (1)** | 10,252 | 95.25% |
| **No Paga (0)** | 511 | 4.75% |

Esto dificulta la detección de la clase minoritaria ("No Paga").

### 3. Variables Pre-Préstamo No Son Suficientes

Con las variables disponibles (solo información previa al préstamo), el modelo logra un **Recall de No Paga de solo 1.96%**. Esto indica que:

> **Se necesitan más variables (historial crediticio, comportamiento pasado) para predecir mejor el impago.**

### 4. Ajuste de Umbral No Mejora la Detección

Probar con umbrales de decisión más bajos (0.4, 0.3, 0.2, 0.1) **no mejoró el Recall de No Paga**, lo que confirma que el problema no es el umbral, sino la falta de información predictiva.

---

## 📈 Monitoreo y Data Drift (Avance #3)

### ¿Qué es Data Drift?

Data Drift ocurre cuando los datos que recibe el modelo en producción **cambian con el tiempo** con respecto a los datos con los que fue entrenado. Esto puede causar que el modelo pierda precisión.

### Métricas de Drift Implementadas

| Métrica | ¿Qué mide? | Umbral de Alerta |
|---------|-----------|------------------|
| **PSI (Population Stability Index)** | Cambio en distribuciones | > 0.1 = moderado, > 0.25 = severo |
| **KS Test** | Diferencia entre distribuciones | p-value < 0.05 |
| **Jensen-Shannon** | Distancia entre distribuciones | > 0.1 |
| **Chi-cuadrado** | Para variables categóricas | p-value < 0.05 |

### Dashboard de Monitoreo (Streamlit)

El dashboard incluye:

| Sección | Función |
|---------|---------|
| **Resumen del Modelo** | Métricas, distribución de clases, matriz de confusión |
| **Predicciones** | Formulario para nuevos clientes y tabla de pronósticos |
| **Monitoreo de Drift** | PSI, KS, Jensen-Shannon, semáforo de alertas |
| **Análisis Temporal** | Evolución del drift en el tiempo |
| **Distribución de Datos** | Comparación de distribuciones históricas vs nuevas |

---

# 🔍 Guía de Revisión del Proyecto MLOps

## 📌 Instrucciones para revisar el proyecto

Esta guía te permitirá ejecutar y probar el proyecto tanto localmente como con Docker. Sigue los pasos en orden para una revisión completa.

---

## 🚀 PARTE 1: REVISIÓN LOCAL (SIN DOCKER)

### 1. Clonar el repositorio

```bash
git clone https://github.com/pachecolanzziano/M5PI.git
cd M5PI

# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Mac/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

#Ejecutar el pipeline de datos
cd src
python ft_engineering.py

# Entrenar y evaluar modelos
python model_training_evaluation.py

# Ejecutar la API localment -> abrir enun navegador: http://localhost:8000/docs
python model_deploy.py

# Ejecutar el dashboard de monitoreo
streamlit run model_monitoring.py
# Probar dashboard: Abrir en navegador: http://localhost:8501

## REVISIÓN CON DOCKER
# Navegar a la raíz del proyecto
cd ..  # Si estás en src/
docker build -t modelo-pago .

# Ejecutar el contenedor
docker run -p 8000:8000 modelo-pago
# Abrir en navegador: http://localhost:8000/docs
#en este endponit podemos probrar la api desde el navegador
# http://localhost:8000/predict


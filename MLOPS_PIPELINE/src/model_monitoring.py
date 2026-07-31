import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from cargar_datos import cargarDatos
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo del Modelo - Avance #3",
    page_icon="📊",
    layout="wide"
)

def convertir_a_string(X):
    """Convierte todas las columnas a tipo string"""
    return X.astype(str)


# ============================================
# 1. CONFIGURACIÓN
# ============================================

MODEL_PATH = "../models/best_model.pkl"
PREPROCESSOR_PATH = "../models/preprocessor.pkl"
DATOS_NUEVOS_PATH = "../Base_de_datos_con_Data_Drift_Simulado.xlsx"

st.title("📊 Aplicación para el Monitoreo de Datos y Modelo")
st.markdown("---")

# ============================================
# 2. CARGAR DATOS
# ============================================

@st.cache_data
def load_data():
    """Carga los datos históricos y los divide en train/test"""
    df = cargarDatos()
    target = "Pago_atiempo"
    X = df.drop(columns=[target])
    y = df[target]
    
    # 🔥 CORRECCIÓN: Asegurar que y sea numérico y binario
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, df

@st.cache_data
def load_new_data():
    """Carga los datos nuevos (con drift simulado)"""
    try:
        df_nuevo = pd.read_excel(DATOS_NUEVOS_PATH)
        return df_nuevo
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model_and_preprocessor():
    """Carga el modelo y el preprocesador entrenados"""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar modelo o preprocesador: {e}")
        return None, None

# Cargar datos
X_train, X_test, y_train, y_test, df_original = load_data()
df_nuevo = load_new_data()
model, preprocessor = load_model_and_preprocessor()

# ============================================
# 3. FUNCIONES DE DRIFT
# ============================================

def calcular_psi(entrenamiento, produccion, bins=10):
    """Population Stability Index (PSI)"""
    psi_resultados = {}
    
    for col in entrenamiento.columns:
        train_vals = entrenamiento[col].dropna().values
        prod_vals = produccion[col].dropna().values
        
        if len(train_vals) == 0 or len(prod_vals) == 0:
            psi_resultados[col] = np.nan
            continue
        
        min_val = min(train_vals.min(), prod_vals.min())
        max_val = max(train_vals.max(), prod_vals.max())
        
        if min_val == max_val:
            psi_resultados[col] = 0.0
            continue
        
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        train_hist, _ = np.histogram(train_vals, bins=bins_edges, density=False)
        prod_hist, _ = np.histogram(prod_vals, bins=bins_edges, density=False)
        
        train_pct = train_hist / train_hist.sum() if train_hist.sum() > 0 else np.zeros_like(train_hist)
        prod_pct = prod_hist / prod_hist.sum() if prod_hist.sum() > 0 else np.zeros_like(prod_hist)
        
        psi = 0
        for t, p in zip(train_pct, prod_pct):
            if t == 0:
                t = 0.001
            if p == 0:
                p = 0.001
            psi += (t - p) * np.log(t / p)
        
        psi_resultados[col] = psi
    
    return psi_resultados


def calcular_ks_test(entrenamiento, produccion):
    """Kolmogorov-Smirnov Test para variables numéricas"""
    ks_resultados = {}
    
    for col in entrenamiento.columns:
        train_vals = entrenamiento[col].dropna().values
        prod_vals = produccion[col].dropna().values
        
        if len(train_vals) == 0 or len(prod_vals) == 0:
            ks_resultados[col] = np.nan
            continue
        
        statistic, pvalue = stats.ks_2samp(train_vals, prod_vals)
        ks_resultados[col] = {
            'statistic': statistic,
            'pvalue': pvalue,
            'drift': statistic > 0.1
        }
    
    return ks_resultados


def calcular_jensen_shannon(entrenamiento, produccion, bins=10):
    """Jensen-Shannon Divergence"""
    js_resultados = {}
    
    for col in entrenamiento.columns:
        train_vals = entrenamiento[col].dropna().values
        prod_vals = produccion[col].dropna().values
        
        if len(train_vals) == 0 or len(prod_vals) == 0:
            js_resultados[col] = np.nan
            continue
        
        min_val = min(train_vals.min(), prod_vals.min())
        max_val = max(train_vals.max(), prod_vals.max())
        
        if min_val == max_val:
            js_resultados[col] = 0.0
            continue
        
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        train_hist, _ = np.histogram(train_vals, bins=bins_edges, density=True)
        prod_hist, _ = np.histogram(prod_vals, bins=bins_edges, density=True)
        
        train_hist = train_hist / train_hist.sum() if train_hist.sum() > 0 else train_hist
        prod_hist = prod_hist / prod_hist.sum() if prod_hist.sum() > 0 else prod_hist
        
        train_hist = np.where(train_hist == 0, 1e-10, train_hist)
        prod_hist = np.where(prod_hist == 0, 1e-10, prod_hist)
        
        js = jensenshannon(train_hist, prod_hist)
        js_resultados[col] = js
    
    return js_resultados


def calcular_chi_cuadrado(entrenamiento, produccion):
    """Chi-cuadrado para variables categóricas"""
    chi2_resultados = {}
    
    for col in entrenamiento.columns:
        train_vals = entrenamiento[col].dropna()
        prod_vals = produccion[col].dropna()
        
        if len(train_vals) == 0 or len(prod_vals) == 0:
            chi2_resultados[col] = np.nan
            continue
        
        # Obtener conteos
        train_counts = train_vals.value_counts()
        prod_counts = prod_vals.value_counts()
        
        # 🔥 CORRECCIÓN: Convertir TODAS las categorías a string
        train_counts.index = train_counts.index.astype(str)
        prod_counts.index = prod_counts.index.astype(str)
        
        all_categories = sorted(set(train_counts.index) | set(prod_counts.index))
        
        train_freq = [train_counts.get(cat, 0) for cat in all_categories]
        prod_freq = [prod_counts.get(cat, 0) for cat in all_categories]
        
        try:
            chi2, pvalue, dof, expected = chi2_contingency([train_freq, prod_freq])
            chi2_resultados[col] = {
                'chi2': chi2,
                'pvalue': pvalue,
                'drift': pvalue < 0.05
            }
        except:
            chi2_resultados[col] = {'chi2': np.nan, 'pvalue': np.nan, 'drift': False}
    
    return chi2_resultados


def interpretar_psi(psi_value):
    """Interpreta el valor de PSI"""
    if psi_value < 0.1:
        return "🟢 Sin drift", "green"
    elif psi_value < 0.25:
        return "🟡 Drift moderado", "orange"
    else:
        return "🔴 Drift severo", "red"


def generar_recomendacion(psi_value, ks_drift, js_value):
    """Genera recomendaciones automáticas basadas en las métricas de drift"""
    recomendaciones = []
    
    if psi_value >= 0.25:
        recomendaciones.append("🔴 **¡ALERTA CRÍTICA!** PSI > 0.25. Se recomienda **retreinar el modelo** de forma inmediata.")
    elif psi_value >= 0.1:
        recomendaciones.append("🟡 **ALERTA MODERADA:** PSI entre 0.1 y 0.25. **Considerar retreinar** el modelo en el corto plazo.")
    else:
        recomendaciones.append("🟢 **MODELO ESTABLE:** PSI < 0.1. No se requiere retreinar.")
    
    if ks_drift:
        recomendaciones.append("📊 **KS Test:** Se detectaron cambios significativos en la distribución de variables. Revisar las variables con drift.")
    
    if js_value > 0.1:
        recomendaciones.append("📊 **Jensen-Shannon:** La distancia entre distribuciones es significativa (> 0.1). Revisar variables afectadas.")
    
    if not recomendaciones:
        recomendaciones.append("✅ Todas las métricas de drift están dentro de rangos aceptables.")
    
    return recomendaciones


def preparar_datos_para_prediccion(df, preprocessor):
    """Prepara los datos para predicción usando el preprocesador"""
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) if target in df.columns else df
    
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    features_used = list(num_features)
    
    features_available = [col for col in features_used if col in X.columns]
    X_prepared = X[features_available]
    X_transformed = preprocessor.transform(X_prepared)
    
    return X_transformed, features_available


def hacer_predicciones(df, model, preprocessor):
    """Hace predicciones sobre un DataFrame"""
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) if target in df.columns else df
    
    X_transformed, features_used = preparar_datos_para_prediccion(df, preprocessor)
    
    y_pred = model.predict(X_transformed)
    y_prob = model.predict_proba(X_transformed)[:, 1]
    
    return y_pred, y_prob, features_used

# ============================================
# 4. SIDEBAR - NAVEGACIÓN
# ============================================

st.sidebar.title("📌 Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Resumen del Modelo",
        "🔮 Predicciones",
        "📈 Monitoreo de Drift",
        "📋 Análisis Temporal",
        "📊 Distribución de Datos"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Avance #3**\n\n"
    "Aplicación para monitorear el modelo y detectar Data Drift.\n\n"
    "**Modelo:** Gradient Boosting\n"
    "**Objetivo:** Predecir Pago_atiempo\n"
    "**Autor:** Luis Carlos Pacheco Lanzziano"
)

# ============================================
# 5. SECCIÓN: RESUMEN DEL MODELO
# ============================================

if opcion == "📊 Resumen del Modelo":
    st.header("📊 Resumen del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelo", "Gradient Boosting")
    
    with col2:
        st.metric("Variables", "10")
    
    with col3:
        st.metric("Target", "Pago_atiempo")
    
    st.markdown("---")
    
    # ============================================
    # DISTRIBUCIÓN DE CLASES
    # ============================================
    
    st.subheader("📊 Distribución de Clases")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Obtener conteos
        class_counts = df_original['Pago_atiempo'].value_counts()
        
        # 🔥 CORRECCIÓN: Reordenar para que Paga (1) aparezca primero
        # y No Paga (0) aparezca después
        class_counts_ordenado = pd.Series({
            'Paga (1)': class_counts.get(1, 0),
            'No Paga (0)': class_counts.get(0, 0)
        })
        
        # Colores: Verde para Paga, Rojo para No Paga
        colors = ['#2ecc71', '#e74c3c']
        
        # Crear barras
        bars = ax.bar(class_counts_ordenado.index, class_counts_ordenado.values, color=colors)
        
        # Títulos y etiquetas
        ax.set_title("Distribución de Clases (Datos Originales)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Pago_atiempo")
        ax.set_ylabel("Cantidad")
        
        # Agregar etiquetas con los valores ENCIMA de las barras
        for bar, valor in zip(bars, class_counts_ordenado.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{valor}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Ajustar límite superior para que las etiquetas no se corten
        ax.set_ylim(0, class_counts_ordenado.values[0] * 1.08)
        
        st.pyplot(fig)
    
    with col2:
        st.write("**Frecuencia por clase:**")
        st.write(f"  - Paga (1): {class_counts.get(1, 0)}")
        st.write(f"  - No Paga (0): {class_counts.get(0, 0)}")
        st.write("")
        
        total = class_counts.sum()
        st.write("**Proporción:**")
        st.write(f"  - Paga (1): {(class_counts.get(1, 0) / total * 100):.2f}%")
        st.write(f"  - No Paga (0): {(class_counts.get(0, 0) / total * 100):.2f}%")
    
    # ============================================
    # MÉTRICAS DEL MODELO
    # ============================================
    
    st.markdown("---")
    st.subheader("📈 Métricas del Modelo")
    
    if model is not None and X_test is not None:
        X_test_transformed, _ = preparar_datos_para_prediccion(X_test, preprocessor)
        y_pred_test = model.predict(X_test_transformed)
        y_prob_test = model.predict_proba(X_test_transformed)[:, 1]
        
        # Asegurar que y_test y y_pred_test sean arrays de enteros
        y_test_array = np.array(y_test).astype(int)
        y_pred_test_array = np.array(y_pred_test).astype(int)
        
        cm = confusion_matrix(y_test_array, y_pred_test_array)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{(y_pred_test == y_test).mean():.2%}")
        with col2:
            recall_clase0 = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
            st.metric("Recall No Paga", f"{recall_clase0:.2%}")
        with col3:
            recall_clase1 = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
            st.metric("Recall Paga", f"{recall_clase1:.2%}")
        with col4:
            f1 = (2 * cm[0][0]) / (2 * cm[0][0] + cm[0][1] + cm[1][0]) if (2 * cm[0][0] + cm[0][1] + cm[1][0]) > 0 else 0
            st.metric("F1-Score", f"{f1:.2%}")
        
        # Matriz de confusión
        st.write("**Matriz de Confusión:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**VP (No Paga detectados):** {cm[0][0]}")
            st.write(f"**FP (Falsos No Paga):** {cm[0][1]}")
        with col2:
            st.write(f"**FN (No Paga no detectados):** {cm[1][0]}")
            st.write(f"**VN (Paga detectados):** {cm[1][1]}")

# ============================================
# 6. SECCIÓN: PREDICCIONES
# ============================================

elif opcion == "🔮 Predicciones":
    st.header("🔮 Predicción de Nuevos Clientes")
    
    st.info("Ingresa los datos del cliente para predecir si pagará o no a tiempo.")
    
    with st.form("prediccion_form"):
        st.subheader("📋 Datos del Cliente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            capital_prestado = st.number_input("Capital Prestado", min_value=0.0, value=3000000.0)
            plazo_meses = st.number_input("Plazo (meses)", min_value=1, max_value=60, value=12)
            edad_cliente = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=35)
            salario_cliente = st.number_input("Salario del Cliente", min_value=0.0, value=2000000.0)
            total_otros_prestamos = st.number_input("Total Otros Préstamos", min_value=0.0, value=500000.0)
        
        with col2:
            cuota_pactada = st.number_input("Cuota Pactada", min_value=0.0, value=250000.0)
            creditos_sectorFinanciero = st.number_input("Créditos Sector Financiero", min_value=0, value=2)
            creditos_sectorCooperativo = st.number_input("Créditos Sector Cooperativo", min_value=0, value=1)
            creditos_sectorReal = st.number_input("Créditos Sector Real", min_value=0, value=0)
            promedio_ingresos_datacredito = st.number_input("Promedio Ingresos DataCrédito", min_value=0.0, value=1800000.0)
        
        submitted = st.form_submit_button("🔮 Predecir")
    
    if submitted:
        if model is None or preprocessor is None:
            st.error("❌ No se pudo cargar el modelo o el preprocesador.")
        else:
            datos_usuario = pd.DataFrame({
                'capital_prestado': [capital_prestado],
                'plazo_meses': [plazo_meses],
                'edad_cliente': [edad_cliente],
                'salario_cliente': [salario_cliente],
                'total_otros_prestamos': [total_otros_prestamos],
                'cuota_pactada': [cuota_pactada],
                'creditos_sectorFinanciero': [creditos_sectorFinanciero],
                'creditos_sectorCooperativo': [creditos_sectorCooperativo],
                'creditos_sectorReal': [creditos_sectorReal],
                'promedio_ingresos_datacredito': [promedio_ingresos_datacredito]
            })
            
            X_transformed, _ = preparar_datos_para_prediccion(datos_usuario, preprocessor)
            y_pred = model.predict(X_transformed)[0]
            y_prob = model.predict_proba(X_transformed)[0][1]
            
            st.markdown("---")
            st.subheader("📊 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if y_pred == 1:
                    st.success("✅ **Paga a tiempo**")
                else:
                    st.error("❌ **No paga a tiempo**")
            
            with col2:
                st.metric("Probabilidad de Pago", f"{y_prob:.2%}")
            
            with col3:
                st.metric("Probabilidad de No Pago", f"{(1-y_prob):.2%}")
            
            st.progress(y_prob)
    
    # Mostrar predicciones en batch (tabla con datos + pronósticos)
    st.markdown("---")
    st.subheader("📋 Tabla de Datos con Pronósticos")
    
    if df_nuevo is not None and model is not None:
        df_con_predicciones = df_nuevo.copy()
        y_pred, y_prob, _ = hacer_predicciones(df_nuevo, model, preprocessor)
        df_con_predicciones['Prediccion'] = y_pred
        df_con_predicciones['Probabilidad_Pago'] = y_prob
        df_con_predicciones['Clasificacion'] = df_con_predicciones['Prediccion'].map({1: '✅ Paga', 0: '❌ No Paga'})
        
        st.dataframe(df_con_predicciones.head(50))
        st.caption(f"📌 Mostrando {min(50, len(df_con_predicciones))} de {len(df_con_predicciones)} registros")
    else:
        st.warning("⚠️ No hay datos nuevos para mostrar predicciones.")

# ============================================
# 7. SECCIÓN: MONITOREO DE DRIFT
# ============================================

elif opcion == "📈 Monitoreo de Drift":
    st.header("📈 Monitoreo de Data Drift")
    
    if df_nuevo is None:
        st.warning("⚠️ No se encontró el archivo de datos nuevos con drift simulado.")
        st.info("Por favor, asegúrate de tener el archivo 'Base_de_datos_con_Data_Drift_Simulado.xlsx' en la raíz del proyecto.")
    else:
        st.success("✅ Datos nuevos cargados correctamente para comparar.")
        
        # Información de los datasets
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datos de Entrenamiento", f"{len(X_train)} registros")
        with col2:
            st.metric("Datos Nuevos (Producción)", f"{len(df_nuevo)} registros")
        
        st.markdown("---")
        
        # Variables numéricas comunes
        variables_numericas = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        variables_numericas = [col for col in variables_numericas if col in df_nuevo.columns and col != 'Pago_atiempo']
        
        variables_categoricas = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        variables_categoricas = [col for col in variables_categoricas if col in df_nuevo.columns and col != 'Pago_atiempo']
        
        # ============================================
        # MÉTRICAS DE DRIFT
        # ============================================
        
        st.subheader("📊 Métricas de Data Drift")
        
        # Calcular todas las métricas
        psi_resultados = calcular_psi(X_train[variables_numericas], df_nuevo[variables_numericas])
        ks_resultados = calcular_ks_test(X_train[variables_numericas], df_nuevo[variables_numericas])
        js_resultados = calcular_jensen_shannon(X_train[variables_numericas], df_nuevo[variables_numericas])
        
        # Crear tabla resumen
        df_drift = pd.DataFrame({
            'Variable': list(psi_resultados.keys()),
            'PSI': [psi_resultados.get(v, np.nan) for v in psi_resultados.keys()],
            'KS_Statistic': [ks_resultados.get(v, {}).get('statistic', np.nan) for v in psi_resultados.keys()],
            'KS_Drift': [ks_resultados.get(v, {}).get('drift', False) for v in psi_resultados.keys()],
            'JS_Divergence': [js_resultados.get(v, np.nan) for v in psi_resultados.keys()]
        })
        
        df_drift['PSI_Interpretacion'] = df_drift['PSI'].apply(
            lambda x: 'Sin drift' if pd.isna(x) else ('Sin drift' if x < 0.1 else ('Drift moderado' if x < 0.25 else 'Drift severo'))
        )
        
        df_drift['KS_Interpretacion'] = df_drift['KS_Drift'].apply(
            lambda x: '⚠️ Drift detectado' if x else '✅ Sin drift'
        )
        
        df_drift['JS_Interpretacion'] = df_drift['JS_Divergence'].apply(
            lambda x: '⚠️ > 0.1' if not pd.isna(x) and x > 0.1 else '✅ < 0.1'
        )
        
        # Mostrar tabla
        st.dataframe(df_drift.style.background_gradient(subset=['PSI'], cmap='RdYlGn_r'))
        
        # ============================================
        # INDICADORES VISUALES DE ALERTA (SEMÁFORO)
        # ============================================
        
        st.markdown("---")
        st.subheader("🚦 Indicadores de Alerta")
        
        col1, col2, col3 = st.columns(3)
        
        # Contar variables por estado de PSI
        psi_sin_drift = len(df_drift[df_drift['PSI_Interpretacion'] == 'Sin drift'])
        psi_moderado = len(df_drift[df_drift['PSI_Interpretacion'] == 'Drift moderado'])
        psi_severo = len(df_drift[df_drift['PSI_Interpretacion'] == 'Drift severo'])
        
        with col1:
            st.metric("🟢 Sin Drift", psi_sin_drift)
        with col2:
            st.metric("🟡 Drift Moderado", psi_moderado)
        with col3:
            st.metric("🔴 Drift Severo", psi_severo)
        
        # ============================================
        # RECOMENDACIONES AUTOMÁTICAS
        # ============================================
        
        st.markdown("---")
        st.subheader("💡 Recomendaciones Automáticas")
        
        # Calcular métricas promedio
        psi_promedio = np.nanmean(df_drift['PSI'].values)
        ks_drift_detectado = df_drift['KS_Drift'].any()
        js_promedio = np.nanmean(df_drift['JS_Divergence'].values)
        
        recomendaciones = generar_recomendacion(psi_promedio, ks_drift_detectado, js_promedio)
        
        for rec in recomendaciones:
            st.info(rec)
        
        # ============================================
        # GRÁFICO DE BARRAS DE PSI
        # ============================================
        
        st.markdown("---")
        st.subheader("📊 PSI por Variable")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x < 0.1 else 'orange' if x < 0.25 else 'red' for x in df_drift['PSI']]
        
        df_drift_sorted = df_drift.sort_values('PSI', ascending=False)
        bars = ax.bar(df_drift_sorted['Variable'], df_drift_sorted['PSI'], color=colors)
        
        ax.axhline(y=0.1, color='green', linestyle='--', label='Umbral Drift Moderado (0.1)')
        ax.axhline(y=0.25, color='red', linestyle='--', label='Umbral Drift Severo (0.25)')
        ax.set_title("PSI por Variable")
        ax.set_ylabel("PSI")
        ax.set_xlabel("Variable")
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        # ============================================
        # CHI-CUADRADO PARA VARIABLES CATEGÓRICAS
        # ============================================
        
        if variables_categoricas:
            st.markdown("---")
            st.subheader("📊 Chi-cuadrado para Variables Categóricas")
            
            chi2_resultados = calcular_chi_cuadrado(
                X_train[variables_categoricas], 
                df_nuevo[variables_categoricas]
            )
            
            df_chi2 = pd.DataFrame({
                'Variable': list(chi2_resultados.keys()),
                'Chi2': [chi2_resultados.get(v, {}).get('chi2', np.nan) for v in chi2_resultados.keys()],
                'P-value': [chi2_resultados.get(v, {}).get('pvalue', np.nan) for v in chi2_resultados.keys()],
                'Drift Detectado': ['⚠️ Sí' if chi2_resultados.get(v, {}).get('drift', False) else '✅ No' 
                                   for v in chi2_resultados.keys()]
            })
            
            st.dataframe(df_chi2)

# ============================================
# 8. SECCIÓN: ANÁLISIS TEMPORAL
# ============================================

elif opcion == "📋 Análisis Temporal":
    st.header("📋 Análisis Temporal del Drift")
    
    if df_nuevo is None:
        st.warning("⚠️ No se encontraron datos nuevos para analizar evolución temporal.")
    else:
        st.info("📌 Simulación de evolución del drift a lo largo del tiempo.")
        
        # Simular fechas para los datos nuevos
        fechas = [datetime.now() - timedelta(days=x*30) for x in range(len(df_nuevo)//100 + 1)]
        fechas = fechas[:len(df_nuevo)//100 + 1]
        
        # Crear batches de datos por fecha
        batch_size = 100
        num_batches = len(df_nuevo) // batch_size + 1
        
        variables_numericas = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        variables_numericas = [col for col in variables_numericas if col in df_nuevo.columns and col != 'Pago_atiempo']
        
        # Calcular PSI por batch
        psi_evolucion = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df_nuevo))
            
            if start_idx >= len(df_nuevo):
                break
            
            batch = df_nuevo.iloc[start_idx:end_idx]
            
            if len(batch) > 10:
                psi_batch = calcular_psi(X_train[variables_numericas], batch[variables_numericas])
                psi_promedio = np.nanmean(list(psi_batch.values()))
                
                psi_evolucion.append({
                    'Batch': i + 1,
                    'Fecha': fechas[i] if i < len(fechas) else datetime.now(),
                    'PSI_Promedio': psi_promedio,
                    'Registros': len(batch)
                })
        
        if psi_evolucion:
            df_evolucion = pd.DataFrame(psi_evolucion)
            
            # Gráfico de evolución
            st.subheader("📈 Evolución del PSI Promedio en el Tiempo")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_evolucion['Fecha'], df_evolucion['PSI_Promedio'], marker='o', linewidth=2)
            ax.axhline(y=0.1, color='green', linestyle='--', label='Umbral Drift Moderado (0.1)')
            ax.axhline(y=0.25, color='red', linestyle='--', label='Umbral Drift Severo (0.25)')
            ax.set_title("Evolución del PSI Promedio en el Tiempo")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("PSI Promedio")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Detectar tendencias
            st.subheader("📊 Análisis de Tendencias")
            
            # Últimos valores
            ultimo_psi = df_evolucion['PSI_Promedio'].iloc[-1]
            primer_psi = df_evolucion['PSI_Promedio'].iloc[0]
            tendencia = "⬆️ Aumentando" if ultimo_psi > primer_psi else "⬇️ Disminuyendo"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("PSI Inicial", f"{primer_psi:.4f}")
            with col2:
                st.metric("PSI Actual", f"{ultimo_psi:.4f}")
            with col3:
                st.metric("Tendencia", tendencia)
            
            # Alertas por cambios abruptos
            st.subheader("🚨 Alertas de Cambios Abruptos")
            
            cambios_abruptos = []
            for i in range(1, len(df_evolucion)):
                cambio = abs(df_evolucion['PSI_Promedio'].iloc[i] - df_evolucion['PSI_Promedio'].iloc[i-1])
                if cambio > 0.05:
                    cambios_abruptos.append({
                        'Fecha': df_evolucion['Fecha'].iloc[i],
                        'Cambio': cambio,
                        'PSI_Anterior': df_evolucion['PSI_Promedio'].iloc[i-1],
                        'PSI_Actual': df_evolucion['PSI_Promedio'].iloc[i]
                    })
            
            if cambios_abruptos:
                for alerta in cambios_abruptos:
                    st.warning(f"🔴 **Cambio abrupto detectado el {alerta['Fecha'].strftime('%Y-%m-%d')}**")
                    st.write(f"   PSI pasó de {alerta['PSI_Anterior']:.4f} a {alerta['PSI_Actual']:.4f} (cambio de {alerta['Cambio']:.4f})")
            else:
                st.success("✅ No se detectaron cambios abruptos significativos.")
            
            st.dataframe(df_evolucion)
        else:
            st.warning("⚠️ No hay suficientes datos para análisis temporal.")

# ============================================
# 9. SECCIÓN: DISTRIBUCIÓN DE DATOS
# ============================================

else:
    st.header("📋 Distribución de Datos")
    
    st.subheader("📊 Datos de Entrenamiento")
    st.write(f"**Dimensiones:** {X_train.shape}")
    
    # 🔥 CORRECCIÓN: Limpiar datos antes de mostrar
    X_train_clean = X_train.copy()
    for col in X_train_clean.columns:
        if X_train_clean[col].dtype == 'object':
            X_train_clean[col] = X_train_clean[col].astype(str)
    st.dataframe(X_train_clean.head())
    
    st.subheader("📊 Datos de Producción (Nuevos)")
    if df_nuevo is not None:
        st.write(f"**Dimensiones:** {df_nuevo.shape}")
        
        # 🔥 CORRECCIÓN: Limpiar datos nuevos antes de mostrar
        df_nuevo_clean = df_nuevo.copy()
        for col in df_nuevo_clean.columns:
            if df_nuevo_clean[col].dtype == 'object':
                df_nuevo_clean[col] = df_nuevo_clean[col].astype(str)
        st.dataframe(df_nuevo_clean.head())
        
        st.subheader("📊 Comparación de Distribuciones")
        variables_numericas = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        variables_numericas = [col for col in variables_numericas if col in df_nuevo.columns]
        
        if variables_numericas:
            # Selector de variable
            var_seleccionada = st.selectbox(
                "Selecciona una variable para comparar:",
                variables_numericas
            )
            
            if var_seleccionada:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    X_train[var_seleccionada].hist(bins=30, ax=ax, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_title(f"Original: {var_seleccionada}", fontsize=12, fontweight='bold')
                    ax.set_xlabel(var_seleccionada, fontsize=10)
                    ax.set_ylabel("Frecuencia", fontsize=10)
                    st.pyplot(fig)
                    
                    st.write("**Estadísticas Originales:**")
                    st.write(X_train[var_seleccionada].describe())
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df_nuevo[var_seleccionada].hist(bins=30, ax=ax, alpha=0.7, color='orange', edgecolor='black')
                    ax.set_title(f"Nuevos: {var_seleccionada}", fontsize=12, fontweight='bold')
                    ax.set_xlabel(var_seleccionada, fontsize=10)
                    ax.set_ylabel("Frecuencia", fontsize=10)
                    st.pyplot(fig)
                    
                    st.write("**Estadísticas Nuevas:**")
                    st.write(df_nuevo[var_seleccionada].describe())
                
                # Gráfico superpuesto
                st.subheader("📊 Comparación Superpuesta")
                fig, ax = plt.subplots(figsize=(10, 5))
                X_train[var_seleccionada].hist(bins=30, ax=ax, alpha=0.5, color='blue', label='Original', edgecolor='black')
                df_nuevo[var_seleccionada].hist(bins=30, ax=ax, alpha=0.5, color='orange', label='Nuevos', edgecolor='black')
                ax.set_title(f"Comparación: {var_seleccionada}", fontsize=14, fontweight='bold')
                ax.set_xlabel(var_seleccionada, fontsize=12)
                ax.set_ylabel("Frecuencia", fontsize=12)
                ax.legend(fontsize=12)
                st.pyplot(fig)
        else:
            st.info("ℹ️ No hay variables numéricas comunes para comparar.")
    
    else:
        st.warning("⚠️ No se encontraron datos nuevos para comparar.")
    
    st.markdown("---")
    st.caption("📌 Los datos de entrenamiento se utilizan como referencia para detectar cambios en los datos de producción.")
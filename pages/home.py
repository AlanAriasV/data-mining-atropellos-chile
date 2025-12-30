import streamlit as st

# Header principal
st.title("🚗 Análisis Predictivo de Atropellos en Chile")
st.markdown("### 📊 Sistema de Machine Learning para Predicción de Gravedad")

st.divider()

# Presentación del proyecto
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 👋 Bienvenido al Sistema de Análisis
    
    Este proyecto utiliza **Machine Learning** para analizar y predecir la gravedad de atropellos 
    en Chile basándose en datos históricos del período 2020-2024.
    
    ### 🎯 Objetivos del Proyecto:
    
    - **Análisis Exploratorio:** Identificar patrones temporales y geográficos en los incidentes
    - **Preprocesamiento Inteligente:** Dos flujos optimizados (Nominal y Numérico)
    - **Modelos Predictivos:** CatBoost para datos categóricos y Random Forest/Decision Tree para datos numéricos
    - **Predicción en Tiempo Real:** Interfaz interactiva para evaluar nuevos casos
    
    ### 📈 Resultados Clave:
    
    - ✅ **+25,000 incidentes** analizados
    - ✅ **70%+ de precisión** en predicción de gravedad
    - ✅ **Identificación de factores** de riesgo más importantes
    - ✅ **Modelos interpretables** para toma de decisiones
    """)

with col2:
    st.info("""
    ### 📚 Tecnologías Utilizadas
    
    **Machine Learning:**
    - CatBoost
    - Random Forest
    - Decision Trees
    - SMOTE (balanceo)
    
    **Análisis de Datos:**
    - Pandas
    - NumPy
    - Scikit-learn
    
    **Visualización:**
    - Streamlit
    - Matplotlib
    - Seaborn
    """)

st.divider()

# Quick Start Guide
st.markdown("## 🚀 Quick Start - Navegación Rápida")

st.write("Selecciona una sección para comenzar tu análisis:")

# Crear grid de 2x2 para las tarjetas
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔍 Análisis Exploratorio (EDA)
    
    Explora el dataset completo con visualizaciones interactivas:
    - 📊 Estadísticas descriptivas
    - 📅 Patrones temporales (años, meses, días, horas)
    - 🗺️ Distribución geográfica
    - 📈 Tendencias y estacionalidad
    """)

    # Enlace a EDA
    if st.button("📊 Ir a EDA", key="btn_eda", use_container_width=True):
        st.switch_page("pages/initial_eda.py")

    st.divider()

    st.markdown("""
    ### 🤖 Modelos de Predicción
    
    Entrena, evalúa y usa modelos de ML:
    - 🌳 **CatBoost** (datos nominales)
    - 🌲 **Random Forest** (datos numéricos)
    - 📊 Métricas de rendimiento
    - 🔮 Predicción en vivo
    """)

    # Enlace a Modelos
    if st.button("🤖 Ir a Modelos", key="btn_models", use_container_width=True):
        st.switch_page("pages/models/models.py")

with col2:
    st.markdown("""
    ### ⚙️ Preprocesamiento de Datos
    
    Visualiza el flujo de preparación de datos:
    - 🎯 Creación de variable target
    - 🔢 Encoding de variables (Nominal/Numérico)
    - ✂️ División train/test estratificada
    - ⚖️ Balanceo de clases (SMOTE/Weights)
    """)

    # Enlace a Preprocesamiento
    if st.button("⚙️ Ir a Preprocesamiento", key="btn_prep", use_container_width=True):
        st.switch_page("pages/preprocessing/preprocessing.py")

    st.divider()

    st.markdown("""
    ### 📖 Flujo de Trabajo Recomendado
    
    1. **EDA** → Entender los datos
    2. **Preprocesamiento** → Ver transformaciones
    3. **Modelos** → Entrenar y predecir
    
    💡 **Tip:** Cada sección tiene código reproducible
    """)

st.divider()

# Información adicional
st.markdown("## 📋 Información del Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📅 Período",
        value="2020-2024",
        delta="5 años"
    )

with col2:
    st.metric(
        label="📊 Incidentes",
        value="25,284",
        delta="Total registrados"
    )

with col3:
    st.metric(
        label="🎯 Target",
        value="Binario",
        delta="Leve/Grave"
    )

with col4:
    st.metric(
        label="🔢 Features",
        value="11",
        delta="Variables predictoras"
    )

st.divider()

# Comparación de modelos
st.markdown("## 🏆 Comparación de Enfoques")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.success("""
    ### 🌳 Enfoque Nominal (CatBoost)
    
    **Ventajas:**
    - ✅ Maneja texto directamente
    - ✅ No requiere encoding
    - ✅ Usa class weights
    - ✅ Interpretable
    
    **Ideal para:**
    - Modelos que preservan semántica
    - Datos categóricos nativos
    - Explicabilidad del negocio
    """)

with comparison_col2:
    st.info("""
    ### 🌲 Enfoque Numérico (RF/Tree)
    
    **Ventajas:**
    - ✅ Compatible con Scikit-learn
    - ✅ Usa SMOTE para balanceo
    - ✅ Múltiples algoritmos
    - ✅ Rápido entrenamiento
    
    **Ideal para:**
    - Experimentación rápida
    - Ensemble methods
    - Pipelines estándar
    """)

st.divider()

# Footer
st.markdown("""
---
### 💡 Notas Importantes

- **Datos Reales:** Este proyecto utiliza datos oficiales de atropellos en Chile
- **Propósito Educativo:** Desarrollado como proyecto de Data Mining
- **Código Abierto:** Todo el código es reproducible y está documentado
- **Actualización:** Los modelos pueden reentrenarse con nuevos datos

### 🔗 Enlaces Rápidos

Usa los botones en la parte superior para navegar entre las diferentes secciones de la aplicación.
""")

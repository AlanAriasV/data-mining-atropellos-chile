import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


@st.cache_data
def load_datasets():
    """Carga los datasets preprocesados nominales"""
    train_df = pd.read_csv("./csv/preprocessed/Train_Binario_Nominal.csv")
    test_df = pd.read_csv("./csv/preprocessed/Test_Binario_Nominal.csv")

    X_train = train_df.drop('Gravedad', axis=1)
    y_train = train_df['Gravedad']
    X_test = test_df.drop('Gravedad', axis=1)
    y_test = test_df['Gravedad']

    return X_train, X_test, y_train, y_test


def load_model(filename):
    """Carga un modelo guardado"""
    try:
        return joblib.load(f"./models/nominal/{filename}")
    except:
        return None


def _tab1_content(X_train, y_train, cb_model):
    st.subheader('🌳 Entrenamiento del Modelo CatBoost')

    st.write("""
    CatBoost es un algoritmo de gradient boosting optimizado para **datos categóricos**.
    Maneja automáticamente variables de texto sin necesidad de encoding previo.
    """)

    # Calcular class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {0: 1.0, 1: 1.6}

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **🎯 Características del modelo:**
        - Iterations: 2000 (con early stopping)
        - Learning rate: 0.03
        - Depth: 8
        - Loss function: Logloss
        - Manejo nativo de categorías
        """)

        st.metric("Iteraciones máximas", "2000")
        st.metric("Profundidad", "8 niveles")

    with col2:
        st.success("""
        **⚖️ Class Weights (Pesos de Clase):**
        - Leve (0): Peso = 1.0
        - Grave (1): Peso = 1.6
        
        Los pesos compensan el desbalance de clases sin generar datos sintéticos.
        """)

        st.metric("Learning Rate", "0.03")
        st.metric("Early Stopping", "100 rounds")

    st.divider()

    # Variables categóricas
    cat_features = ['Dia_semana', 'Zona', 'Tipo_Calza',
                    'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']

    st.write('**📊 Información del entrenamiento:**')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features totales", X_train.shape[1])
    with col2:
        st.metric("Features categóricos", len(cat_features))
    with col3:
        balance = y_train.value_counts(normalize=True) * 100
        st.metric("Balance de clases",
                  f"{balance[0]:.1f}% / {balance[1]:.1f}%")

    st.write('**Variables categóricas manejadas nativamente:**')
    cat_list = ', '.join([f'`{col}`' for col in cat_features])
    st.markdown(cat_list)

    with st.expander("📝 Ver código de entrenamiento"):
        st.code("""from catboost import CatBoostClassifier

cat_features = ['Dia_semana', 'Zona', 'Tipo_Calza', 
                'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    class_weights={0: 1.0, 1: 1.6},
    cat_features=cat_features,
    verbose=100,
    early_stopping_rounds=100,
    random_strength=2,
    l2_leaf_reg=10
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))""", language='python')


def _tab2_content(cb_model, X_test, y_test):
    st.subheader('📈 Evaluación del Modelo')

    st.write("""
    Evaluación del rendimiento de CatBoost en el **conjunto de prueba** (datos reales).
    """)

    # Predicciones
    y_pred = cb_model.predict(X_test)

    # Métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

    # Mostrar métricas
    st.write('**🎯 Métricas de rendimiento:**')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

    st.divider()

    # Matriz de confusión
    st.write('**🔍 Matriz de Confusión:**')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Leve (0)', 'Grave (1)'],
                yticklabels=['Leve (0)', 'Grave (1)'])
    ax.set_title('Matriz de Confusión - CatBoost',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Verdad (Realidad)', fontsize=12)
    ax.set_xlabel('Predicción', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Reporte detallado
    st.write('**📋 Reporte de Clasificación:**')
    report = classification_report(y_test, y_pred, target_names=[
                                   'Leve (0)', 'Grave (1)'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    st.success(f"""
    ✅ El modelo CatBoost alcanzó una exactitud de **{metrics['Accuracy']:.2%}** en el conjunto de prueba.
    """)

    with st.expander("📝 Ver código de evaluación"):
        st.code("""from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Leve (0)', 'Grave (1)'])

print(f"Accuracy: {accuracy:.4f}")
print(report)""", language='python')


def _tab3_content(cb_model, X_train):
    st.subheader('📊 Importancia de Variables')

    st.write("""
    CatBoost calcula automáticamente la importancia de cada variable para la predicción.
    Mayor importancia = mayor influencia en las decisiones del modelo.
    """)

    # Obtener importancias
    importance = cb_model.get_feature_importance()
    features = X_train.columns

    importance_df = pd.DataFrame({
        'Variable': features,
        'Importancia': importance
    }).sort_values('Importancia', ascending=False)

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_idx = np.argsort(importance)

    ax.barh(range(len(sorted_idx)), importance[sorted_idx],
            color=sns.color_palette('viridis', len(sorted_idx)))
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([features[i] for i in sorted_idx])
    ax.set_title('Importancia de Variables - CatBoost',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_ylabel('Variable', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Tabla de importancia
    st.write('**🏆 Ranking completo de importancia:**')
    st.dataframe(importance_df.reset_index(drop=True),
                 hide_index=True, use_container_width=True)

    # Insights
    top_var = importance_df.iloc[0]['Variable']
    top_importance = importance_df.iloc[0]['Importancia']

    st.info(f"""
    💡 **Insight:** La variable más importante es `{top_var}` con una importancia de **{top_importance:.2f}**.
    Esto significa que esta variable tiene el mayor impacto en las predicciones del modelo.
    """)

    with st.expander("📝 Ver código"):
        st.code("""# Obtener importancia de variables
importance = model.get_feature_importance()
features = X_train.columns

# Crear DataFrame y ordenar
df_importance = pd.DataFrame({
    'Variable': features,
    'Importancia': importance
}).sort_values('Importancia', ascending=False)

# Visualizar
plt.barh(features, importance)""", language='python')


def _tab4_content(cb_model, X_train):
    st.subheader('🔮 Predicción en Vivo')
    st.write("""
    Ingresa los datos de un incidente para predecir su gravedad usando el modelo **CatBoost**.
    """)

    st.info("""
    💡 **Ventaja:** CatBoost trabaja directamente con valores categóricos (texto), 
    por lo que puedes ver exactamente qué estás seleccionando.
    """)

    # Cargar datos originales para obtener las opciones
    from utils.load_data import load_data
    df_original = load_data()

    # Crear formulario
    with st.form("prediction_form_nominal"):
        st.markdown("### 📅 Variables Temporales")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            año = st.number_input("Año", min_value=2020,
                                  max_value=2030, value=2024)
        with col2:
            mes = st.selectbox("Mes", options=list(range(1, 13)),
                               format_func=lambda x: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                                                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][x-1])
        with col3:
            # Día de la semana con valores originales
            dias_semana = sorted(df_original['Dia_semana'].unique())
            dia_semana = st.selectbox("Día de la semana", options=dias_semana)
        with col4:
            hora = st.number_input(
                "Hora aproximada", min_value=0, max_value=23, value=12)

        st.markdown("### 📍 Variables de Ubicación")
        col1, col2 = st.columns(2)

        with col1:
            regions = {
                'Tarapacá': 1, 'Antofagasta': 2, 'Atacama': 3, 'Coquimbo': 4,
                'Valparaíso': 5, "Libertador General Bernardo O'Higgins": 6,
                'Maule': 7, 'Biobío': 8, 'La Araucanía': 9, 'Los Lagos': 10,
                'Aysén del General Carlos Ibáñez del Campo': 11,
                'Magallanes y de la Antártica Chilena': 12,
                'Región Metropolitana': 13, 'Los Ríos': 14,
                'Arica y Parinacota': 15, 'Ñuble': 16
            }
            cut_reg_text = st.selectbox(
                "CUT_REG (Región)", options=regions.keys())
            cut_reg = regions[cut_reg_text]

        with col2:
            # Zona con valores originales
            zonas = sorted(df_original['Zona'].unique())
            zona = st.selectbox("Zona", options=zonas)

        st.markdown("### 🌍 Variables de Entorno")
        col1, col2 = st.columns(2)

        with col1:
            # Tipo de Calzada
            tipos_calza = sorted(df_original['Tipo_Calza'].unique())
            tipo_calza = st.selectbox("Tipo de Calzada", options=tipos_calza)

            # Estado de Calzada
            estados_cal = sorted(df_original['Estado_Cal'].unique())
            estado_cal = st.selectbox("Estado de Calzada", options=estados_cal)

        with col2:
            # Condición
            condiciones = sorted(df_original['Condición'].unique())[1:]
            condicion = st.selectbox("Condición", options=condiciones)

            # Estado Atmosférico
            estados_atm = sorted(df_original['Estado_Atm'].unique())
            estado_atm = st.selectbox(
                "Estado Atmosférico", options=estados_atm)

        st.markdown("### 🚨 Causa")
        # Causa
        causas = sorted(df_original['Causa__CON'].unique())
        causa_con = st.selectbox("Causa del Accidente", options=causas)

        # Botón de predicción
        submitted = st.form_submit_button(
            "🔮 Predecir Gravedad", use_container_width=True)

    if submitted:
        # Crear DataFrame con los datos ingresados (valores originales, no codificados)
        input_data = pd.DataFrame({
            'Año': [año],
            'Mes': [mes],
            'Dia_semana': [dia_semana],
            'Hora_aprox': [hora],
            'CUT_REG': [cut_reg],
            'Zona': [zona],
            'Tipo_Calza': [tipo_calza],
            'Estado_Cal': [estado_cal],
            'Condición': [condicion],
            'Estado_Atm': [estado_atm],
            'Causa__CON': [causa_con]
        })

        print(f'input_data: {input_data}')

        # Asegurar que las columnas estén en el mismo orden que el entrenamiento
        input_data = input_data[X_train.columns]

        # Hacer predicción
        prediction = cb_model.predict(input_data)[0]
        prediction_proba = cb_model.predict_proba(input_data)[0]

        st.divider()

        # Mostrar resultado
        st.markdown("### 🎯 Resultado de la Predicción")
        st.caption("🤖 Modelo utilizado: **CatBoost (Nominal)**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicción", '✅ LEVE' if prediction ==
                      0 else '⚠️ GRAVE')
        with col2:
            st.metric("Probabilidad Leve", f"{prediction_proba[0]:.2%}")
        with col3:
            st.metric("Probabilidad Grave", f"{prediction_proba[1]:.2%}")

        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#28a745' if prediction == 0 else '#6c757d',
                  '#dc3545' if prediction == 1 else '#6c757d']
        bars = ax.barh(['Leve (0)', 'Grave (1)'],
                       prediction_proba, color=colors)
        ax.set_xlabel('Probabilidad', fontsize=12)
        ax.set_title('Distribución de Probabilidades',
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)

        # Agregar valores
        for i, v in enumerate(prediction_proba):
            ax.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("""
        💡 **Interpretación:** CatBoost procesó directamente los valores categóricos sin necesidad de encoding.
        La clase con mayor probabilidad es la predicción final.
        """)


def main():
    st.title('🤖 Modelo de Clasificación - Nominal (CatBoost)')

    st.write("""
    Entrenamiento, evaluación y predicción con **CatBoost**, un algoritmo optimizado 
    para datos categóricos que no requiere encoding previo.
    """)

    # Cargar datos
    with st.spinner('Cargando datasets...'):
        X_train, X_test, y_train, y_test = load_datasets()

    # Cargar modelo
    with st.spinner('Cargando modelo CatBoost...'):
        cb_model = load_model("cb_model.joblib")

    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Entrenamiento",
        "📈 Evaluación",
        "📊 Importancia de Variables",
        "🔮 Predicción en Vivo"
    ])

    # TAB 1: Entrenamiento
    with tab1:
        _tab1_content(X_train, y_train, cb_model)

    # TAB 2: Evaluación
    with tab2:
        _tab2_content(cb_model, X_test, y_test)

    # TAB 3: Importancia de Variables
    with tab3:
        _tab3_content(cb_model, X_train)

    # TAB 4: Predicción en Vivo
    with tab4:
        _tab4_content(cb_model, X_train)

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_datasets():
    """Carga los datasets preprocesados"""
    train_df = pd.read_csv("./csv/preprocessed/Train_Numerico_SMOTE.csv")
    test_df = pd.read_csv("./csv/preprocessed/Test_Numerico.csv")

    X_train = train_df.drop('Gravedad', axis=1)
    y_train = train_df['Gravedad']
    X_test = test_df.drop('Gravedad', axis=1)
    y_test = test_df['Gravedad']

    return X_train, X_test, y_train, y_test


def load_model(filename):
    """Carga un modelo guardado"""
    try:
        return joblib.load(f"./models/{filename}")
    except:
        return None


def _tab1_content(X_train, y_train, tree_model, rf_model):
    st.subheader('🌳 Entrenamiento de Modelos')

    st.write("""
    Se entrenan **dos modelos** de clasificación basados en árboles de decisión 
    para predecir la gravedad de los atropellos.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **🌲 Árbol de Decisión**
        - Modelo simple y explicable
        - `max_depth=10` para evitar overfitting
        - Reglas claras y visualizables
        - Ideal para interpretabilidad
        """)

        st.metric("Profundidad máxima", "10 niveles")
        st.metric("Datos de entrenamiento", f"{len(X_train):,}")

    with col2:
        st.success("""
        **🌳 Random Forest**
        - Ensemble de 100 árboles
        - `max_depth=10` por árbol
        - Mayor precisión que árbol único
        - Reduce overfitting
        """)

        st.metric("Número de árboles", "100")
        st.metric("Profundidad máxima", "10 niveles")

    st.divider()

    # Información de entrenamiento
    st.write('**📊 Información del entrenamiento:**')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features utilizados", X_train.shape[1])
    with col2:
        st.metric("Muestras de entrenamiento", f"{len(X_train):,}")
    with col3:
        balance = y_train.value_counts(normalize=True) * 100
        st.metric("Balance de clases",
                  f"{balance[0]:.1f}% / {balance[1]:.1f}%")

    st.write('**Variables utilizadas:**')
    features_list = ', '.join([f'`{col}`' for col in X_train.columns])
    st.markdown(features_list)

    with st.expander("📝 Ver código de entrenamiento"):
        st.code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Árbol de Decisión
tree_model = DecisionTreeClassifier(random_state=42, max_depth=10)
tree_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)""", language='python')


def _tab2_content(tree_model, rf_model, X_test, y_test):
    st.subheader('📈 Evaluación de Modelos')

    st.write("""
    Comparación del rendimiento de ambos modelos en el **conjunto de prueba** (datos reales sin SMOTE).
    """)

    # Predicciones
    y_pred_tree = tree_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    # Métricas
    metrics_tree = {
        'Accuracy': accuracy_score(y_test, y_pred_tree),
        'Precision': precision_score(y_test, y_pred_tree),
        'Recall': recall_score(y_test, y_pred_tree),
        'F1-Score': f1_score(y_test, y_pred_tree)
    }

    metrics_rf = {
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-Score': f1_score(y_test, y_pred_rf)
    }

    # Mostrar métricas comparativas
    st.write('**🎯 Métricas de rendimiento:**')

    # Crear grid de 3 columnas por métrica
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.markdown(f"**Métrica**")
    with col2:
        st.markdown("**🌲 Árbol de Decisión**")
    with col3:
        st.markdown("**🌳 Random Forest**")

    for metric in metrics_tree.keys():
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.metric(f"**{metric}**", f"{metric}",
                      label_visibility='collapsed')
        with col2:
            tree_val = metrics_tree[metric]
            st.metric("🌲 Árbol", f"{tree_val:.4f}",
                      label_visibility='collapsed')
        with col3:
            rf_val = metrics_rf[metric]
            delta = rf_val - tree_val
            # Mostrar valor y diferencia como texto
            st.metric("🌳 Random Forest",
                      f"{rf_val:.4f}", f"{delta:.4f}", label_visibility='collapsed')

    st.divider()

    # Matrices de confusión
    st.write('**🔍 Matrices de Confusión:**')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Árbol de Decisión
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Leve (0)', 'Grave (1)'],
                yticklabels=['Leve (0)', 'Grave (1)'])
    ax1.set_title('Árbol de Decisión', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Verdad (Realidad)', fontsize=10)
    ax1.set_xlabel('Predicción', fontsize=10)

    # Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Leve (0)', 'Grave (1)'],
                yticklabels=['Leve (0)', 'Grave (1)'])
    ax2.set_title('Random Forest', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Verdad (Realidad)', fontsize=10)
    ax2.set_xlabel('Predicción', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Conclusión
    if metrics_rf['Accuracy'] > metrics_tree['Accuracy']:
        st.success(f"""
        ✅ **Random Forest** tiene mejor desempeño general con una exactitud de **{metrics_rf['Accuracy']:.2%}** 
        vs **{metrics_tree['Accuracy']:.2%}** del Árbol de Decisión.
        """)
    else:
        st.success(f"""
        ✅ **Árbol de Decisión** tiene mejor desempeño general con una exactitud de **{metrics_tree['Accuracy']:.2%}** 
        vs **{metrics_rf['Accuracy']:.2%}** del Random Forest.
        """)

    with st.expander("📝 Ver código de evaluación"):
        st.code("""from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(report)""", language='python')


def _tab3_content(tree_model, rf_model, X_train):
    st.subheader('📊 Importancia de Variables')

    st.write("""
    Ranking de las variables más importantes para la predicción según cada modelo.
    Mayor importancia = mayor influencia en la decisión del modelo.
    """)

    # Obtener importancias
    tree_importance = pd.DataFrame({
        'Variable': X_train.columns,
        'Importancia': tree_model.feature_importances_
    }).sort_values('Importancia', ascending=False)

    rf_importance = pd.DataFrame({
        'Variable': X_train.columns,
        'Importancia': rf_model.feature_importances_
    }).sort_values('Importancia', ascending=False)

    # Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Árbol de Decisión
    sns.barplot(data=tree_importance, x='Importancia', y='Variable',
                hue='Variable', palette='viridis', ax=ax1, legend=False)
    ax1.set_title('Importancia - Árbol de Decisión',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Importancia', fontsize=10)
    ax1.set_ylabel('Variable', fontsize=10)

    # Random Forest
    sns.barplot(data=rf_importance, x='Importancia', y='Variable',
                hue='Variable', palette='rocket', ax=ax2, legend=False)
    ax2.set_title('Importancia - Random Forest',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Importancia', fontsize=10)
    ax2.set_ylabel('Variable', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Tablas de importancia
    col1, col2 = st.columns(2)

    with col1:
        st.write('**🌲 Top 5 - Árbol de Decisión:**')
        st.dataframe(tree_importance.head(
            5).reset_index(drop=True), hide_index=True)

    with col2:
        st.write('**🌳 Top 5 - Random Forest:**')
        st.dataframe(rf_importance.head(
            5).reset_index(drop=True), hide_index=True)

    # Insights
    top_var_tree = tree_importance.iloc[0]['Variable']
    top_var_rf = rf_importance.iloc[0]['Variable']

    st.info(f"""
    💡 **Insights:**
    - **Árbol de Decisión:** La variable más importante es `{top_var_tree}`
    - **Random Forest:** La variable más importante es `{top_var_rf}`
    """)

    with st.expander("📝 Ver código"):
        st.code("""# Obtener importancia de variables
importances = model.feature_importances_
features = X_train.columns

# Crear DataFrame y ordenar
df_importance = pd.DataFrame({
    'Variable': features,
    'Importancia': importances
}).sort_values('Importancia', ascending=False)

# Visualizar
sns.barplot(data=df_importance, x='Importancia', y='Variable')""", language='python')


def _tab4_content(tree_model, rf_model, X_train):
    st.subheader('🔮 Predicción en Vivo')
    st.write("""
    Ingresa los datos de un incidente para predecir su gravedad usando modelos de Machine Learning.
    """)

    # Selector de modelo
    modelo_seleccionado = st.selectbox(
        "🤖 Selecciona el modelo a usar:",
        options=["Random Forest", "Árbol de Decisión"],
        help="Ambos modelos fueron entrenados con los mismos datos"
    )

    # Seleccionar el modelo según la elección
    modelo_actual = rf_model if modelo_seleccionado == "Random Forest" else tree_model

    st.info("""
    💡 **Instrucciones:** Selecciona los valores para cada variable y presiona "Predecir" para obtener el resultado.
    """)
    # Cargar los label encoders desde el dataset original
    from utils.load_data import load_data
    df_original = load_data()
    # Crear los encoders con los datos originales
    label_encoders = {}
    cols_texto = ['Dia_semana', 'Zona', 'Tipo_Calza',
                  'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']
    for col in cols_texto:
        le = LabelEncoder()
        le.fit(df_original[col].astype(str))
        label_encoders[col] = le

    # Crear formulario
    with st.form("prediction_form"):
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
            dias_semana = sorted(label_encoders['Dia_semana'].classes_)
            dia_semana_text = st.selectbox(
                "Día de la semana", options=dias_semana)
            dia_semana = label_encoders['Dia_semana'].transform([dia_semana_text])[
                0]

        with col4:
            hora = st.number_input(
                "Hora aproximada", min_value=0, max_value=23, value=12)

        st.markdown("### 📍 Variables de Ubicación")

        col1, col2 = st.columns(2)

        with col1:
            regions = {
                'Tarapacá': 1,
                'Antofagasta': 2,
                'Atacama': 3,
                'Coquimbo': 4,
                'Valparaíso': 5,
                "Libertador General Bernardo O'Higgins": 6,
                'Maule': 7,
                'Biobío': 8,
                'La Araucanía': 9,
                'Los Lagos': 10,
                'Aysén del General Carlos Ibáñez del Campo': 11,
                'Magallanes y de la Antártica Chilena': 12,
                'Región Metropolitana': 13,
                'Los Ríos': 14,
                'Arica y Parinacota': 15,
                'Ñuble': 16
            }
            cut_reg_text = st.selectbox(
                "CUT_REG (Región)", options=regions.keys())
            cut_reg = regions[cut_reg_text]

        with col2:
            # Zona con valores originales
            zonas = sorted(label_encoders['Zona'].classes_)
            zona_text = st.selectbox("Zona", options=zonas)
            zona = label_encoders['Zona'].transform([zona_text])[0]

        st.markdown("### 🌍 Variables de Entorno")

        col1, col2 = st.columns(2)

        with col1:
            # Tipo de Calzada con valores originales
            tipos_calza = sorted(label_encoders['Tipo_Calza'].classes_)
            tipo_calza_text = st.selectbox(
                "Tipo de Calzada", options=tipos_calza)
            tipo_calza = label_encoders['Tipo_Calza'].transform([tipo_calza_text])[
                0]
            # Estado de Calzada con valores originales
            estados_cal = sorted(label_encoders['Estado_Cal'].classes_)
            estado_cal_text = st.selectbox(
                "Estado de Calzada", options=estados_cal)
            estado_cal = label_encoders['Estado_Cal'].transform([estado_cal_text])[
                0]
        with col2:
            # Condición con valores originales
            condiciones = sorted(label_encoders['Condición'].classes_)[1:]

            condicion_text = st.selectbox("Condición", options=condiciones)
            condicion = label_encoders['Condición'].transform([condicion_text])[
                0]

            # Estado Atmosférico con valores originales
            estados_atm = sorted(label_encoders['Estado_Atm'].classes_)
            estado_atm_text = st.selectbox(
                "Estado Atmosférico", options=estados_atm)
            estado_atm = label_encoders['Estado_Atm'].transform([estado_atm_text])[
                0]
        st.markdown("### 🚨 Causa")
        # Causa con valores originales
        causas = sorted(label_encoders['Causa__CON'].classes_)
        causa_con_text = st.selectbox("Causa del Accidente", options=causas)
        causa_con = label_encoders['Causa__CON'].transform([causa_con_text])[0]
        # Botón de predicción
        submitted = st.form_submit_button(
            "🔮 Predecir Gravedad", use_container_width=True)
    if submitted:
        # Crear DataFrame con los datos ingresados
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

        # Asegurar que las columnas estén en el mismo orden que el entrenamiento
        input_data = input_data[X_train.columns]
        # Hacer predicción con el modelo seleccionado
        prediction = modelo_actual.predict(input_data)[0]
        prediction_proba = modelo_actual.predict_proba(input_data)[0]
        st.divider()
        # Mostrar resultado
        st.markdown("### 🎯 Resultado de la Predicción")
        st.caption(f"🤖 Modelo utilizado: **{modelo_seleccionado}**")
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
        💡 **Interpretación:** El modelo asigna probabilidades a cada clase. 
        La clase con mayor probabilidad es la predicción final.
        """)


def main():
    st.title('🤖 Modelos de Clasificación - Numérico')

    st.write("""
    Entrenamiento, evaluación y predicción con modelos de **Árbol de Decisión** y **Random Forest** 
    para clasificar la gravedad de atropellos.
    """)

    # Cargar datos
    with st.spinner('Cargando datasets...'):
        X_train, X_test, y_train, y_test = load_datasets()

    # Entrenar modelos
    with st.spinner('Cargando modelos...'):
        tree_model = load_model("tree_model.joblib")
        rf_model = load_model("rf_model.joblib")

    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Entrenamiento",
        "📈 Evaluación",
        "📊 Importancia de Variables",
        "🔮 Predicción en Vivo"
    ])

    # TAB 1: Entrenamiento
    with tab1:
        _tab1_content(X_train, y_train, tree_model, rf_model)

    # TAB 2: Evaluación
    with tab2:
        _tab2_content(tree_model, rf_model, X_test, y_test)

    # TAB 3: Importancia de Variables
    with tab3:
        _tab3_content(tree_model, rf_model, X_train)

    # TAB 4: Predicción en Vivo
    with tab4:
        _tab4_content(tree_model, rf_model, X_train)

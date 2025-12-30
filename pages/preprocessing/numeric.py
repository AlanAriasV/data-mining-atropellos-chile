import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.load_data import load_data


@st.cache_data
def _create_target(df):
    df_copy = df.copy()
    df_copy['Gravedad'] = ((df_copy['Fallecidos'] > 0) |
                           (df_copy['Graves'] > 0)).astype(int)
    return df_copy


@st.cache_data
def _create_numeric_dataset(df):
    """Crea dataset numérico con LabelEncoding de variables categóricas"""
    columnas_input = [
        'Año', 'Mes', 'Dia_semana', 'Hora_aprox',  # Temporales
        'CUT_REG', 'Zona',                         # Ubicación
        'Tipo_Calza', 'Estado_Cal', 'Condición', 'Estado_Atm',  # Entorno
        'Causa__CON'                               # Causa
    ]
    target = 'Gravedad'

    df_numeric = df[columnas_input + [target]].copy()

    # Mapear meses a números
    mapa_meses = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    df_numeric['Mes'] = df_numeric['Mes'].map(mapa_meses)

    # Codificar variables categóricas
    cols_texto = ['Dia_semana', 'Zona', 'Tipo_Calza',
                  'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']

    label_encoders = {}
    for col in cols_texto:
        le = LabelEncoder()
        df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
        label_encoders[col] = le

    return df_numeric, label_encoders


def _tab1_content(df_with_target):
    st.subheader('🎯 Creación del Target Binario')

    st.write("""
    Se crea una variable objetivo binaria **Gravedad** basada en la severidad del incidente:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Clase 1 (Grave)**  
        - Fallecidos > 0 **O**  
        - Graves > 0
        """)
    with col2:
        st.success("""
        **Clase 0 (Leve)**  
        - Menos Graves  
        - Leves  
        - Ilesos
        """)

    st.write('**Distribución de clases:**')

    # Gráfico de distribución
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df_with_target['Gravedad'].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
    ax.set_title('Distribución de Gravedad Binaria (Original)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Gravedad (0=Leve, 1=Grave)', fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=12)
    ax.set_xticklabels(['Leve (0)', 'Grave (1)'])

    # Agregar valores en las barras
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    st.pyplot(fig)
    plt.close()

    # Estadísticas
    st.write('**Estadísticas:**')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total de incidentes', f"{len(df_with_target):,}")
    with col2:
        clase_0 = (df_with_target['Gravedad'] == 0).sum()
        pct_0 = (clase_0 / len(df_with_target) * 100)
        st.metric('Leve (0)', f"{clase_0:,}", f"{pct_0:.1f}%")
    with col3:
        clase_1 = (df_with_target['Gravedad'] == 1).sum()
        pct_1 = (clase_1 / len(df_with_target) * 100)
        st.metric('Grave (1)', f"{clase_1:,}", f"{pct_1:.1f}%")

    st.warning(f"""
    ⚠️ **Desbalance de clases:** La clase minoritaria (Grave) representa solo el {pct_1:.1f}% de los datos.
    Para el flujo numérico, este desbalance se maneja con **SMOTE** (Synthetic Minority Over-sampling Technique).
    """)

    with st.expander("📝 Ver código"):
        st.code(
            """df['Gravedad'] = ((df['Fallecidos'] > 0) | (df['Graves'] > 0)).astype(int)""", language='python')


def _tab2_content(df_numeric, label_encoders):
    st.subheader('🔢 Codificación de Variables')

    st.write("""
    Se transforman las variables categóricas a **valores numéricos** mediante **Label Encoding** 
    para que puedan ser procesadas por algoritmos de scikit-learn y SMOTE.
    """)

    st.info("""
    📌 **Label Encoding:** Asigna un número único a cada categoría de texto.  
    Ejemplo: "Lunes" → 0, "Martes" → 1, "Miércoles" → 2, etc.
    """)

    # Mostrar variables codificadas
    st.write('**Variables codificadas:**')
    cols_texto = ['Dia_semana', 'Zona', 'Tipo_Calza',
                  'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📅 Variables Temporales:**
        - `Dia_semana` *(codificado)*
        
        **📍 Variables de Ubicación:**
        - `Zona` *(codificado)*
        """)

    with col2:
        st.markdown("""
        **🌍 Variables de Entorno:**
        - `Tipo_Calza` *(codificado)*
        - `Estado_Cal` *(codificado)*
        - `Condición` *(codificado)*
        - `Estado_Atm` *(codificado)*
        - `Causa__CON` *(codificado)*
        """)

    st.divider()

    # Mostrar ejemplo de codificación
    st.write('**Ejemplo de codificación (Zona):**')
    if 'Zona' in label_encoders:
        zona_mapping = pd.DataFrame({
            'Categoría Original': label_encoders['Zona'].classes_,
            'Código Numérico': range(len(label_encoders['Zona'].classes_))
        })
        st.dataframe(zona_mapping, hide_index=True)

    st.divider()

    st.write('**Vista previa del dataset numérico:**')
    st.dataframe(df_numeric.head(10), hide_index=True)

    st.write('**Información del dataset:**')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Filas', f"{df_numeric.shape[0]:,}")
    with col2:
        st.metric('Columnas (features + target)', df_numeric.shape[1])

    st.success("""
    ✅ **Ventaja:** Todos los datos son numéricos, compatibles con scikit-learn y SMOTE.
    """)

    with st.expander("📝 Ver código"):
        st.code("""from sklearn.preprocessing import LabelEncoder

cols_texto = ['Dia_semana', 'Zona', 'Tipo_Calza', 
              'Estado_Cal', 'Condición', 'Estado_Atm', 'Causa__CON']

label_encoders = {}
for col in cols_texto:
    le = LabelEncoder()
    df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    label_encoders[col] = le""", language='python')


def _tab3_content(df_numeric, X_train, X_test, y_train, y_test, X_train_smote, y_train_smote):
    st.subheader('✂️ División Train/Test + SMOTE')

    st.write("""
    Se divide el dataset en conjuntos de **entrenamiento (80%)** y **prueba (20%)** 
    con estratificación, y luego se aplica **SMOTE** solo al conjunto de entrenamiento.
    """)

    st.info("""
    🎯 **SMOTE (Synthetic Minority Over-sampling Technique):**  
    Genera ejemplos sintéticos de la clase minoritaria para balancear el dataset de entrenamiento.
    **Importante:** Solo se aplica al Train Set para evitar data leakage.
    """)

    # Métricas ANTES de SMOTE
    st.write('**📊 Antes de SMOTE:**')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total de datos', f"{len(df_numeric):,}")
    with col2:
        st.metric('Train (80%)', f"{len(X_train):,}")
        train_dist_before = y_train.value_counts(normalize=True) * 100
        st.caption(
            f"Leve: {train_dist_before[0]:.1f}% | Grave: {train_dist_before[1]:.1f}%")
    with col3:
        st.metric('Test (20%)', f"{len(X_test):,}")
        test_dist = y_test.value_counts(normalize=True) * 100
        st.caption(f"Leve: {test_dist[0]:.1f}% | Grave: {test_dist[1]:.1f}%")

    st.divider()

    # Métricas DESPUÉS de SMOTE
    st.write('**🔄 Después de SMOTE (solo Train):**')
    col1, col2 = st.columns(2)

    with col1:
        st.metric('Train con SMOTE', f"{len(X_train_smote):,}",
                  f"+{len(X_train_smote) - len(X_train):,} sintéticos")
        train_dist_after = y_train_smote.value_counts(normalize=True) * 100
        st.caption(
            f"Leve: {train_dist_after[0]:.1f}% | Grave: {train_dist_after[1]:.1f}%")

    with col2:
        st.metric('Test (sin cambios)', f"{len(X_test):,}")
        st.caption("⚠️ El Test Set NO se modifica (datos reales)")

    # Gráfico comparativo
    st.write('**Comparación visual:**')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Antes de SMOTE
    counts_before = y_train.value_counts().sort_index()
    sns.barplot(x=counts_before.index, y=counts_before.values,
                palette='viridis', ax=ax1)
    ax1.set_title('Train ANTES de SMOTE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Gravedad', fontsize=10)
    ax1.set_ylabel('Cantidad', fontsize=10)
    ax1.set_xticklabels(['Leve (0)', 'Grave (1)'])
    for i, v in enumerate(counts_before.values):
        ax1.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    # Después de SMOTE
    counts_after = y_train_smote.value_counts().sort_index()
    sns.barplot(x=counts_after.index, y=counts_after.values,
                palette='rocket', ax=ax2)
    ax2.set_title('Train DESPUÉS de SMOTE', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Gravedad', fontsize=10)
    ax2.set_ylabel('Cantidad', fontsize=10)
    ax2.set_xticklabels(['Leve (0)', 'Grave (1)'])
    for i, v in enumerate(counts_after.values):
        ax2.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success("""
    ✅ **Resultado:** Las clases están balanceadas en el Train Set, mejorando el aprendizaje del modelo.
    """)

    with st.expander("📝 Ver código"):
        st.code("""from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aplicar SMOTE solo al Train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)""", language='python')


def _tab4_content(X_train_smote, X_test, y_train_smote, y_test):
    st.subheader('💾 Datasets Finales Generados')

    st.write("""
    El preprocesamiento numérico genera **2 archivos CSV** listos para entrenar modelos 
    de scikit-learn (Random Forest, XGBoost, etc.).
    """)

    # Información de archivos
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **📁 Train_Numerico_SMOTE.csv**
        - Datos de entrenamiento con SMOTE
        - Todas las variables numéricas
        - Clases balanceadas (50/50)
        - Incluye datos sintéticos
        - Para entrenar el modelo
        """)

        st.metric('Filas en Train', f"{len(X_train_smote):,}")
        st.metric('Columnas', f"{X_train_smote.shape[1] + 1}")  # +1 por target

    with col2:
        st.success("""
        **📁 Test_Numerico.csv**
        - Datos de prueba (SIN SMOTE)
        - Todas las variables numéricas
        - Distribución real de clases
        - Solo datos originales
        - Para validar el modelo
        """)

        st.metric('Filas en Test', f"{len(X_test):,}")
        st.metric('Columnas', f"{X_test.shape[1] + 1}")  # +1 por target

    st.divider()

    st.write('**Vista previa de Train Set (con SMOTE):**')
    train_preview = pd.concat([X_train_smote, y_train_smote], axis=1)
    st.dataframe(train_preview.head(10), hide_index=True)

    st.write('**Vista previa de Test Set (sin SMOTE):**')
    test_preview = pd.concat([X_test, y_test], axis=1)
    st.dataframe(test_preview.head(10), hide_index=True)

    st.divider()

    st.write('**📊 Características del preprocesamiento numérico:**')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ Ventajas:**
        - Compatible con scikit-learn
        - Clases balanceadas con SMOTE
        - Datos completamente numéricos
        - Ideal para Random Forest, XGBoost
        - Mejora el aprendizaje de clase minoritaria
        """)

    with col2:
        st.markdown("""
        **⚙️ Uso recomendado:**
        - Modelos: Random Forest, XGBoost, SVM
        - Manejo de desbalance: SMOTE
        - Requiere encoding previo
        - Bueno para modelos tradicionales
        """)

    with st.expander("📝 Ver código"):
        st.code("""# Guardar datasets numéricos
train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)
test_numeric = pd.concat([X_test, y_test], axis=1)

train_smote.to_csv("./csv_preprocess/Train_Numerico_SMOTE.csv", index=False)
test_numeric.to_csv("./csv_preprocess/Test_Numerico.csv", index=False)""", language='python')


def main():
    st.title('🔢 Preprocesamiento Numérico')

    st.write("""
    Este flujo de preprocesamiento **codifica todas las variables categóricas a números** 
    y aplica **SMOTE** para balancear las clases. Ideal para modelos de scikit-learn 
    como **Random Forest**, **XGBoost**, o **SVM**.
    """)

    # Cargar datos
    df = load_data()

    # Crear todos los datasets procesados UNA SOLA VEZ
    df_with_target = _create_target(df)
    df_numeric, label_encoders = _create_numeric_dataset(df_with_target)

    # División Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        df_numeric.drop('Gravedad', axis=1),
        df_numeric['Gravedad'],
        test_size=0.2,
        random_state=42,
        stratify=df_numeric['Gravedad']
    )

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Creación del Target",
        "🔢 Codificación",
        "✂️ División + SMOTE",
        "💾 Datasets Finales"
    ])

    # TAB 1: Creación del Target
    with tab1:
        _tab1_content(df_with_target)

    # TAB 2: Codificación
    with tab2:
        _tab2_content(df_numeric, label_encoders)

    # TAB 3: División + SMOTE
    with tab3:
        _tab3_content(df_numeric, X_train, X_test, y_train,
                      y_test, X_train_smote, y_train_smote)

    # TAB 4: Datasets Finales
    with tab4:
        _tab4_content(X_train_smote, X_test, y_train_smote, y_test)

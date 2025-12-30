import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_data


@st.cache_data
def _create_target(df):
    df_copy = df.copy()
    df_copy['Gravedad'] = ((df_copy['Fallecidos'] > 0) |
                           (df_copy['Graves'] > 0)).astype(int)
    return df_copy


@st.cache_data
def _create_nominal_dataset(df):
    columnas_input = [
        'Año', 'Mes', 'Dia_semana', 'Hora_aprox',  # Temporales
        'CUT_REG', 'Zona',                         # Ubicación
        'Tipo_Calza', 'Estado_Cal', 'Condición', 'Estado_Atm',  # Entorno
        'Causa__CON'                               # Causa
    ]
    target = 'Gravedad'

    df_nominal = df[columnas_input + [target]].copy()

    # Mapear meses a números para orden lógico
    mapa_meses = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    df_nominal['Mes'] = df_nominal['Mes'].map(mapa_meses)

    return df_nominal


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
    Para el flujo nominal, este desbalance se maneja con **pesos de clase** internos en CatBoost.
    """)

    with st.expander("📝 Ver código"):
        st.code(
            """df['Gravedad'] = ((df['Fallecidos'] > 0) | (df['Graves'] > 0)).astype(int)""", language='python')


def _tab2_content(df_nominal):
    st.subheader('📋 Selección de Atributos')

    st.write("""
    Se seleccionan las variables más relevantes para predecir la gravedad del atropello.
    Las variables categóricas se **mantienen en texto** para aprovechar las capacidades nativas de CatBoost.
    """)

    # Mostrar columnas seleccionadas por categoría
    col1, col2 = st.columns(2)

    with col1:
        st.write('**⏰ Variables Temporales:**')
        st.markdown("""
        - `Año`
        - `Mes` *(mapeado a números 1-12)*
        - `Dia_semana`
        - `Hora_aprox`
        """)

        st.write('**📍 Variables de Ubicación:**')
        st.markdown("""
        - `CUT_REG` (Región)
        - `Zona`
        """)

    with col2:
        st.write('**🌍 Variables de Entorno:**')
        st.markdown("""
        - `Tipo_Calza` (Tipo de calzada)
        - `Estado_Cal` (Estado de la calzada)
        - `Condición` (Condición climática)
        - `Estado_Atm` (Estado atmosférico)
        """)

        st.write('**🚨 Variable de Causa:**')
        st.markdown("""
        - `Causa__CON`
        """)

    st.divider()

    st.write('**Vista previa del dataset nominal:**')
    st.dataframe(df_nominal.head(10), use_container_width=True)

    st.write('**Información del dataset:**')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Filas', f"{df_nominal.shape[0]:,}")
    with col2:
        st.metric('Columnas (features + target)', df_nominal.shape[1])

    # Mostrar tipos de datos
    st.write('**Tipos de datos:**')
    tipos_df = pd.DataFrame({
        'Columna': df_nominal.columns,
        'Tipo': df_nominal.dtypes.values,
        'Ejemplo': [df_nominal[col].iloc[0] for col in df_nominal.columns]
    })
    st.dataframe(tipos_df, hide_index=True, use_container_width=True)

    with st.expander("📝 Ver código"):
        st.code("""columnas_input = [
    'Año', 'Mes', 'Dia_semana', 'Hora_aprox',  # Temporales
    'CUT_REG', 'Zona',                         # Ubicación
    'Tipo_Calza', 'Estado_Cal', 'Condición', 'Estado_Atm',  # Entorno
    'Causa__CON'                               # Causa
]

df_nominal = df[columnas_input + ['Gravedad']].copy()

# Mapear meses a números
mapa_meses = {'Enero': 1, 'Febrero': 2, ..., 'Diciembre': 12}
df_nominal['Mes'] = df_nominal['Mes'].map(mapa_meses)""", language='python')


def _tab3_content(df_nominal, X_train_nom, X_test_nom, y_train_nom, y_test_nom):
    st.subheader('✂️ División Train/Test Estratificada')

    st.write("""
    Se divide el dataset en conjuntos de **entrenamiento (80%)** y **prueba (20%)** 
    manteniendo la proporción de clases mediante **estratificación**.
    """)

    st.info("""
    🎯 **Estratificación:** Garantiza que ambos conjuntos (train y test) tengan la misma 
    proporción de incidentes graves y leves que el dataset original.
    """)

    # Métricas de división
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total de datos', f"{len(df_nominal):,}")
    with col2:
        st.metric('Train (80%)', f"{len(X_train_nom):,}")
    with col3:
        st.metric('Test (20%)', f"{len(X_test_nom):,}")

    st.divider()

    # Comparación de distribuciones
    st.write('**Verificación de estratificación:**')

    col1, col2 = st.columns(2)

    with col1:
        st.write('**Train Set:**')
        train_dist = y_train_nom.value_counts(normalize=True) * 100
        train_df = pd.DataFrame({
            'Clase': ['Leve (0)', 'Grave (1)'],
            'Cantidad': y_train_nom.value_counts().sort_index().values,
            'Porcentaje': [f"{train_dist[0]:.2f}%", f"{train_dist[1]:.2f}%"]
        })
        st.dataframe(train_df, hide_index=True, use_container_width=True)

    with col2:
        st.write('**Test Set:**')
        test_dist = y_test_nom.value_counts(normalize=True) * 100
        test_df = pd.DataFrame({
            'Clase': ['Leve (0)', 'Grave (1)'],
            'Cantidad': y_test_nom.value_counts().sort_index().values,
            'Porcentaje': [f"{test_dist[0]:.2f}%", f"{test_dist[1]:.2f}%"]
        })
        st.dataframe(test_df, hide_index=True, use_container_width=True)

    st.success(
        "✅ Las proporciones se mantienen consistentes entre Train y Test gracias a la estratificación.")

    with st.expander("📝 Ver código"):
        st.code("""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_nominal.drop('Gravedad', axis=1),
    df_nominal['Gravedad'],
    test_size=0.2,
    random_state=42,
    stratify=df_nominal['Gravedad']  # Mantiene proporciones
)""", language='python')


def _tab4_content(X_train_nom, X_test_nom, y_train_nom, y_test_nom):
    st.subheader('💾 Datasets Finales Generados')

    st.write("""
    El preprocesamiento nominal genera **2 archivos CSV** listos para entrenar modelos 
    que manejan datos categóricos nativamente (CatBoost, LightGBM).
    """)

    # Información de archivos
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **📁 Train_Binario_Nominal.csv**
        - Datos de entrenamiento (80%)
        - Variables categóricas en texto
        - Incluye target 'Gravedad'
        - Para entrenar el modelo
        """)

        st.metric('Filas en Train', f"{len(X_train_nom):,}")
        st.metric('Columnas', f"{X_train_nom.shape[1] + 1}")  # +1 por target

    with col2:
        st.success("""
        **📁 Test_Binario_Nominal.csv**
        - Datos de prueba (20%)
        - Variables categóricas en texto
        - Incluye target 'Gravedad'
        - Para validar el modelo
        """)

        st.metric('Filas en Test', f"{len(X_test_nom):,}")
        st.metric('Columnas', f"{X_test_nom.shape[1] + 1}")  # +1 por target

    st.divider()

    st.write('**Vista previa de Train Set:**')
    train_preview = pd.concat([X_train_nom, y_train_nom], axis=1)
    st.dataframe(train_preview.head(10), use_container_width=True)

    st.write('**Vista previa de Test Set:**')
    test_preview = pd.concat([X_test_nom, y_test_nom], axis=1)
    st.dataframe(test_preview.head(10), use_container_width=True)

    st.divider()

    st.write('**📊 Características del preprocesamiento nominal:**')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ Ventajas:**
        - Mantiene información semántica original
        - No requiere encoding manual
        - CatBoost maneja automáticamente categorías
        - Menos riesgo de data leakage
        - Más interpretable
        """)

    with col2:
        st.markdown("""
        **⚙️ Uso recomendado:**
        - Modelos: CatBoost, LightGBM
        - Manejo de desbalance: Pesos de clase
        - No requiere SMOTE
        - Ideal para producción
        """)

    with st.expander("📝 Ver código"):
        st.code("""# Guardar datasets nominales
train_nominal = pd.concat([X_train, y_train], axis=1)
test_nominal = pd.concat([X_test, y_test], axis=1)

train_nominal.to_csv("./csv_preprocess/Train_Binario_Nominal.csv", index=False)
test_nominal.to_csv("./csv_preprocess/Test_Binario_Nominal.csv", index=False)""", language='python')


def main():
    st.title('🔠 Preprocesamiento Nominal')

    st.write("""
    Este flujo de preprocesamiento mantiene las variables categóricas en su **formato de texto original** 
    (ej: "Urbana", "Lunes", etc.) para ser utilizadas con modelos que manejan nativamente datos categóricos 
    como **CatBoost** o **LightGBM**.
    """)

    # Cargar datos
    df = load_data()

    # Crear todos los datasets procesados UNA SOLA VEZ
    df_with_target = _create_target(df)
    df_nominal = _create_nominal_dataset(df_with_target)

    # División Train/Test
    from sklearn.model_selection import train_test_split
    X_train_nom, X_test_nom, y_train_nom, y_test_nom = train_test_split(
        df_nominal.drop('Gravedad', axis=1),
        df_nominal['Gravedad'],
        test_size=0.2,
        random_state=42,
        stratify=df_nominal['Gravedad']
    )

    # Crear tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Creación del Target",
        "📋 Selección de Atributos",
        "✂️ División Train/Test",
        "💾 Datasets Finales"
    ])

    # TAB 1: Creación del Target
    with tab1:
        _tab1_content(df_with_target)

    # TAB 2: Selección de Atributos
    with tab2:
        _tab2_content(df_nominal)

    # TAB 3: División Train/Test
    with tab3:
        _tab3_content(df_nominal, X_train_nom, X_test_nom,
                      y_train_nom, y_test_nom)

    # TAB 4: Datasets Finales
    with tab4:
        _tab4_content(X_train_nom, X_test_nom, y_train_nom, y_test_nom)

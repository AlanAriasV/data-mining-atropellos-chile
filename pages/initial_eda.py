import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_data


@st.cache_data
def get_years_count(df):
    """Calcula el número de años únicos en el dataset (cacheado)"""
    return df['Fecha'].apply(lambda x: pd.to_datetime(x).year).nunique()


def tab1_content():

    st.subheader('📂 Información del Dataset')

    st.info('💡 **Carga de datos:** Este dataset se carga desde un archivo CSV que contiene el registro histórico de atropellos.')

    st.code('df = pd.read_csv("Atropellos_Consolidado_2020_2024.csv")',
            language='python')

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('📊 Total de Incidentes', f"{df.shape[0]:,}")
    with col2:
        st.metric('📋 Variables', df.shape[1])
    with col3:
        años = get_years_count(df)
        st.metric('📅 Años de Datos', años)

    st.divider()

    st.markdown('### 🔍 Estructura del Dataset')
    st.write(
        'Análisis detallado de cada columna: tipo de dato, valores únicos y valores nulos.')

    columnas_df = pd.DataFrame({
        'Columnas': df.columns,
        'Tipo de dato': df.dtypes.values,
        'Valores únicos': [df[col].nunique() for col in df.columns],
        'Valores nulos': df.isnull().sum().values,
    })
    st.dataframe(columnas_df, hide_index=True, use_container_width=True)

    with st.expander("📝 Ver código"):
        st.code("""columnas_df = pd.DataFrame({
    'Columnas': df.columns,
    'Tipo de dato': df.dtypes.values,
    'Valores únicos': [df[col].nunique() for col in df.columns],
    'Valores nulos': df.isnull().sum().values,
})
st.dataframe(columnas_df, hide_index=True, use_container_width=True)""", language='python')

    st.divider()

    st.markdown('### 👁️ Vista Previa de los Datos')
    st.write('Primeras 10 filas del dataset para entender su estructura:')
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📝 Ver código"):
        st.code("""st.dataframe(df.head(10), use_container_width=True)""",
                language='python')


def tab2_content():

    st.subheader('📈 Estadísticas Descriptivas')

    st.markdown('### 🔢 Variables Numéricas')
    st.info('📊 Resumen estadístico de las variables numéricas: media, desviación estándar, mínimo, máximo y cuartiles.')
    st.dataframe(df.describe(), use_container_width=True)

    with st.expander("📝 Ver código"):
        st.code(
            """st.dataframe(df.describe(), use_container_width=True)""", language='python')

    st.divider()

    st.markdown('### 🅰️ Variables Categóricas')
    st.info(
        '📝 Resumen de las variables de texto: frecuencia de aparición y valores únicos.')
    st.dataframe(df.describe(include='object'), use_container_width=True)

    with st.expander("📝 Ver código"):
        st.code(
            """st.dataframe(df.describe(include='object'), use_container_width=True)""", language='python')


def tab3_content():

    st.subheader('📅 Análisis Temporal')

    st.success(
        '🕒 **Objetivo:** Identificar patrones temporales en los atropellos para detectar períodos de mayor riesgo.')

    # Convertir fecha
    df['fecha'] = pd.to_datetime(df['Fecha'])

    st.markdown('### 📆 Evolución Anual')
    st.write('📈 Tendencia de atropellos a lo largo de los años. Permite identificar si hay aumento o disminución en la incidencia.')

    atropellos_año = df['fecha'].dt.year.value_counts().sort_index()

    # Crear gráfico profesional
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=atropellos_año.index, y=atropellos_año.values,
                palette='viridis', ax=ax)
    ax.set_title('Evolución Anual de Atropellos (2020-2024)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Cantidad de Atropellos', fontsize=12)

    # Agregar valores en las barras
    for i, v in enumerate(atropellos_año.values):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📝 Ver código"):
        st.code("""atropellos_año = df['fecha'].dt.year.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=atropellos_año.index, y=atropellos_año.values,
            palette='viridis', ax=ax)
ax.set_title('Evolución Anual de Atropellos (2020-2024)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
st.pyplot(fig)""", language='python')

    st.divider()

    st.markdown('### 📅 Distribución Mensual')
    st.write('🍃 Identifica meses con mayor concentración de incidentes. Útil para detectar estacionalidad.')

    atropellos_mes = df['fecha'].dt.month.value_counts().sort_index()
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    # Crear gráfico profesional
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(atropellos_mes.index, atropellos_mes.values, marker='o', linewidth=2,
            markersize=8, color='#2E86AB', markerfacecolor='#A23B72')
    ax.set_title('Distribución Mensual de Atropellos',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses_nombres)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Agregar valores en los puntos
    for i, v in enumerate(atropellos_mes.values, 1):
        ax.text(i, v + 50, f'{v:,}', ha='center',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📝 Ver código"):
        st.code("""atropellos_mes = df['fecha'].dt.month.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(atropellos_mes.index, atropellos_mes.values, marker='o', linewidth=2)
ax.set_title('Distribución Mensual de Atropellos',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Mes', fontsize=12)
ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
ax.grid(True, alpha=0.3)
st.pyplot(fig)""", language='python')

    st.divider()

    st.markdown('### 🗓️ Patrones Semanales')
    st.write(
        '📆 Determina qué días de la semana presentan mayor riesgo de atropellos.')

    df['dia_semana'] = df['fecha'].dt.day_name()

    # Orden lógico de días (Lunes a Domingo)
    orden_dias = ['Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday', 'Sunday']
    nombres_dias = ['Lunes', 'Martes', 'Miércoles',
                    'Jueves', 'Viernes', 'Sábado', 'Domingo']

    dia_counts = df['dia_semana'].value_counts()
    dia_counts = dia_counts.reindex(orden_dias)

    # Crear gráfico profesional
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(nombres_dias, dia_counts.values,
                   color=sns.color_palette('viridis', 7))
    ax.set_title('Atropellos por Día de la Semana',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Cantidad de Atropellos', fontsize=12)
    ax.set_ylabel('Día de la Semana', fontsize=12)

    # Agregar valores en las barras
    for i, v in enumerate(dia_counts.values):
        ax.text(v + 50, i, f'{v:,}', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📝 Ver código"):
        st.code("""df['dia_semana'] = df['fecha'].dt.day_name()
orden_dias = ['Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday']
dia_counts = df['dia_semana'].value_counts().reindex(orden_dias)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(nombres_dias, dia_counts.values, color=sns.color_palette('viridis', 7))
ax.set_title('Atropellos por Día de la Semana', fontsize=14, fontweight='bold')
st.pyplot(fig)""", language='python')

    st.divider()

    if 'Hora_aprox' in df.columns:
        st.markdown('### ⏰ Distribución Horaria')
        st.write('🌆 Identifica las horas del día con mayor peligrosidad. Clave para estrategias de prevención y control de tráfico.')

        hora_counts = df['Hora_aprox'].value_counts().sort_index()

        # Crear gráfico profesional
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(hora_counts.index, hora_counts.values,
                      color=sns.color_palette('rocket', len(hora_counts)))
        ax.set_title('Distribución Horaria de Atropellos',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Hora del Día', fontsize=12)
        ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
        ax.set_xticks(range(0, 24))
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Agregar valores en las barras más altas
        max_value = hora_counts.max()
        for i, (hora, valor) in enumerate(hora_counts.items()):
            if valor > max_value * 0.7:  # Solo mostrar en barras altas
                ax.text(hora, valor + 20,
                        f'{valor:,}', ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("📝 Ver código"):
            st.code("""hora_counts = df['Hora_aprox'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hora_counts.index, hora_counts.values,
       color=sns.color_palette('rocket', len(hora_counts)))
ax.set_title('Distribución Horaria de Atropellos',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Hora del Día', fontsize=12)
ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
st.pyplot(fig)""", language='python')


def tab4_content():

    st.subheader('🗺️ Análisis Geográfico')

    st.success('📍 **Objetivo:** Identificar las zonas geográficas con mayor concentración de atropellos para priorizar intervenciones de seguridad vial.')

    st.markdown('### 🏙️ Distribución por Tipo de Zona')
    st.write(
        '📊 Comparación de incidentes entre zonas urbanas y rurales (solo 2 categorías).')

    col_ubicacion = 'Zona'
    top_zonas = df[col_ubicacion].value_counts().head(10)

    # Crear gráfico profesional
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_zonas)), top_zonas.values,
                   color=sns.color_palette('mako_r', len(top_zonas)))
    ax.set_yticks(range(len(top_zonas)))
    ax.set_yticklabels(top_zonas.index)
    ax.set_title('Top 10 Zonas con Mayor Concentración de Atropellos',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Cantidad de Atropellos', fontsize=12)
    ax.set_ylabel('Zona', fontsize=12)
    ax.invert_yaxis()  # Mayor valor arriba

    # Agregar valores en las barras
    for i, v in enumerate(top_zonas.values):
        ax.text(v + 50, i, f'{v:,}', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📝 Ver código"):
        st.code("""col_ubicacion = 'Zona'
zonas_counts = df[col_ubicacion].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#FF6B6B' if zona == 'Urbana' else '#4ECDC4' for zona in zonas_counts.index]
ax.bar(zonas_counts.index, zonas_counts.values, color=colors, width=0.6)
ax.set_title('Comparación de Atropellos: Zona Urbana vs Rural', fontsize=14, fontweight='bold')
ax.set_xlabel('Tipo de Zona', fontsize=12)
ax.set_ylabel('Cantidad de Atropellos', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
st.pyplot(fig)""", language='python')

    st.divider()

    st.markdown('### 🌍 Mapa Interactivo de Incidentes')
    st.write("""
    🗺️ Visualización geoespacial de **todos los atropellos registrados**. Cada punto representa un incidente,
    permitiendo identificar patrones de concentración geográfica y zonas de alta peligrosidad.
    """)
    mapa_df = df[['Lat', 'Lon']].dropna().rename(
        columns={'Lat': 'latitude', 'Lon': 'longitude'})
    st.map(mapa_df)

    with st.expander("📝 Ver código"):
        st.code("""mapa_df = df[['Lat', 'Lon']].dropna().rename(
    columns={'Lat': 'latitude', 'Lon': 'longitude'})
st.map(mapa_df)""", language='python')


st.title('🔍 EDA - Análisis Exploratorio Inicial')

st.info("""
📊 **Análisis Exploratorio de Datos de Atropellos (2020-2024)**

Este análisis examina incidentes de atropellos registrados durante un período de 5 años,
con el objetivo de identificar patrones temporales, geográficos y características clave
que permitan desarrollar estrategias efectivas de prevención y seguridad vial.
""")

# Cargar datos una sola vez
df = load_data()

# Crear tabs para organizar el contenido
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Carga de datos",
    "📈 Estadísticas Descriptivas",
    "📅 Análisis Temporal",
    "🗺️ Análisis Geográfico"
])

# TAB 1: Carga de datos
with tab1:
    tab1_content()

# TAB 2: Estadísticas Descriptivas
with tab2:
    tab2_content()

# TAB 3: Análisis Temporal
with tab3:
    tab3_content()

# TAB 4: Análisis Geográfico
with tab4:
    tab4_content()

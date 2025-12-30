# 🚗 Análisis Predictivo de Atropellos en Chile

Sistema de Machine Learning para análisis y predicción de gravedad de atropellos basado en datos históricos 2020-2024.

## 📊 Características

- **Análisis Exploratorio (EDA)**: Visualizaciones interactivas de patrones temporales y geográficos
- **Preprocesamiento Dual**: Flujos optimizados para datos nominales y numéricos
- **Modelos Predictivos**:
  - 🌳 CatBoost (datos categóricos nativos)
  - 🌲 Random Forest / Decision Tree (datos numéricos con SMOTE)
- **Predicción en Tiempo Real**: Interfaz interactiva para evaluar nuevos casos
- **Navegación Moderna**: Barra superior horizontal con navegación por botones

## 🚀 Instalación

### Prerrequisitos
- Python 3.8+
- pip

### Pasos

1. Clona el repositorio:
```bash
git clone https://github.com/TU_USUARIO/data-mining-atropellos-chile.git
cd data-mining-atropellos-chile
```

2. Crea un entorno virtual:
```bash
python -m venv .venv
```

3. Activa el entorno virtual:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
data-mining-atropellos-chile/
├── main.py                      # Punto de entrada principal
├── pages/
│   ├── home.py                  # Página de inicio
│   ├── initial_eda.py           # Análisis exploratorio
│   ├── preprocessing/
│   │   ├── preprocessing.py     # Router de preprocesamiento
│   │   ├── nominal.py           # Preprocesamiento nominal
│   │   └── numeric.py           # Preprocesamiento numérico
│   └── models/
│       ├── models.py            # Router de modelos
│       ├── nominal.py           # Modelo CatBoost
│       └── numeric.py           # Modelos RF/Tree
├── utils/
│   └── load_data.py             # Utilidades de carga de datos
├── csv/
│   └── preprocessed/            # Datasets procesados
├── models/
│   ├── nominal/                 # Modelos Nominales guardados
│   │   └── cb_model.joblib      # Modelo CatBoost
│   └── numeric/                 # Modelos Numéricos guardados
│       ├── rf_model.joblib      # Modelo Random Forest
│       └── tree_model.joblib    # Modelo Decision Tree
├── .streamlit/
│   └── config.toml              # Configuración de tema
└── requirements.txt             # Dependencias del proyecto
```

## 🎮 Uso

1. Ejecuta la aplicación:
```bash
streamlit run main.py
```

2. Abre tu navegador en `http://localhost:8501`

3. Navega por las secciones:
   - **Inicio**: Presentación y quick start
   - **EDA**: Explora los datos
   - **Preprocesamiento**: Visualiza transformaciones (Nominal/Numérico)
   - **Modelos**: Entrena, evalúa y predice (CatBoost/RF-Tree)

## 📈 Resultados

- ✅ **+25,000 incidentes** analizados
- ✅ **70%+ de precisión** en predicción de gravedad
- ✅ **Identificación de factores** de riesgo más importantes
- ✅ **Modelos interpretables** para toma de decisiones

## 🛠️ Tecnologías

**Machine Learning:**
- CatBoost
- Random Forest
- Decision Trees
- SMOTE (balanceo de clases)

**Análisis de Datos:**
- Pandas
- NumPy
- Scikit-learn

**Visualización:**
- Streamlit
- Matplotlib
- Seaborn

## 📝 Notas

- Los datos son reales de atropellos en Chile (2020-2024)
- Proyecto desarrollado con fines educativos
- Los modelos pueden reentrenarse con nuevos datos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👤 Autor

Proyecto de Data Mining - Análisis de Atropellos en Chile

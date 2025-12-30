"""
Descarga automática de modelos desde Google Drive o GitHub Releases
Este script se ejecuta automáticamente al iniciar la app en Streamlit Cloud
"""

import os
import gdown
import streamlit as st

# URLs de los modelos (actualizar con tus URLs reales)
MODEL_URLS = {
    "models/numeric/tree_model.joblib": "https://drive.google.com/file/d/1koPJdMS1-vA8z_Edj9IpkPAIFmjx8wZ1/view?usp=drive_link",
    "models/numeric/rf_model.joblib": "https://drive.google.com/file/d/1yGSEIVXRjFwLq-qxQ4OH5WQ1NEmcd_cd/view?usp=drive_link",
    "models/nominal/cb_model.joblib": "https://drive.google.com/file/d/1xgdEzl8gSbBTceidhizS7bzBDYI1ueuH/view?usp=drive_link"
}


@st.cache_resource
def download_models():
    """Descarga los modelos si no existen localmente"""

    for model_path, url in MODEL_URLS.items():
        if not os.path.exists(model_path):
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            print(f"Descargando {model_path}...")

            # Descargar desde Google Drive
            if "drive.google.com" in url and url != "TU_URL_GOOGLE_DRIVE":
                gdown.download(url, model_path, quiet=False, fuzzy=True)
            else:
                print(f"⚠️  URL no configurada para {model_path}")
                print(f"   Por favor, sube el modelo a Google Drive y actualiza la URL")

    return True


# Ejecutar al importar
if __name__ != "__main__":
    download_models()

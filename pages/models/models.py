import os

import streamlit as st

from pages.models.nominal import main as nominal_main
from pages.models.numeric import main as numeric_main
from download_models import download_models

if not (os.path.exists("models/numeric/rf_model.joblib") and os.path.exists("models/numeric/tree_model.joblib") and os.path.exists("models/nominal/cb_model.joblib")):
    download_models()

# Crear navegación con botones tipo pills
st.markdown("### 🤖 Modelos de Clasificación")

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if st.button("🌳 CatBoost (Nominal)", use_container_width=True, type="primary" if st.session_state.get('model_view', 'nominal') == 'nominal' else "secondary"):
        st.session_state['model_view'] = 'nominal'
        st.rerun()

with col2:
    if st.button("🌲 RF/Tree (Numérico)", use_container_width=True, type="primary" if st.session_state.get('model_view', 'nominal') == 'numeric' else "secondary"):
        st.session_state['model_view'] = 'numeric'
        st.rerun()

st.divider()

# Mostrar contenido según selección
if st.session_state.get('model_view', 'nominal') == 'nominal':
    nominal_main()
else:
    numeric_main()

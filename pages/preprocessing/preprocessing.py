import streamlit as st

from pages.preprocessing.nominal import main as nominal_main
from pages.preprocessing.numeric import main as numeric_main

# Crear navegación con botones tipo pills
st.markdown("### ⚙️ Preprocesamiento de Datos")

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if st.button("📝 Nominal", use_container_width=True, type="primary" if st.session_state.get('prep_view', 'nominal') == 'nominal' else "secondary"):
        st.session_state['prep_view'] = 'nominal'
        st.rerun()

with col2:
    if st.button("🔢 Numérico", use_container_width=True, type="primary" if st.session_state.get('prep_view', 'nominal') == 'numeric' else "secondary"):
        st.session_state['prep_view'] = 'numeric'
        st.rerun()

st.divider()

# Mostrar contenido según selección
if st.session_state.get('prep_view', 'nominal') == 'nominal':
    nominal_main()
else:
    numeric_main()

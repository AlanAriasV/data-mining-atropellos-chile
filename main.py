

def set_page_config():
    """Configurar tema y página"""
    import streamlit as st

    st.set_page_config(
        page_title="Análisis de Atropellos - ML",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def define_pages():
    from streamlit import navigation, Page

    main_page = Page("./pages/home.py", title="Inicio", icon="🏠")
    eda_page = Page("./pages/initial_eda.py",
                    title="EDA", icon="🔍")
    preprocessing_page = Page("./pages/preprocessing/preprocessing.py",
                              title="Preprocesamiento", icon="⚙️")
    models_page = Page("./pages/models/models.py",
                       title="Modelos", icon="🤖")

    pg = navigation(
        [main_page, eda_page, preprocessing_page, models_page],
        position="top"  # Barra superior en lugar de sidebar
    )

    pg.run()


if __name__ == "__main__":

    set_page_config()
    define_pages()

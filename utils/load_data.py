from streamlit import cache_data
from pandas import read_csv


@cache_data
def load_data():
    return read_csv('./csv/Atropellos_Consolidado_2020_2024.csv')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_and_prepare_data, fit_quantile_regression, plot_best_fit, PALETTE

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Regresión Cuantílica",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("Análisis de Regresión Cuantílica")
st.markdown("""
Esta aplicación permite analizar datos temporales utilizando regresión cuantílica.
Seleccione una columna para ver su análisis y el mejor modelo de ajuste.
""")

# Cargar datos
try:
    df = load_and_prepare_data("data/Seguimiento de datos.xlsx")
    
    # Obtener columnas numéricas
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Sidebar para selección de parámetros
    st.sidebar.header("Parámetros de Análisis")
    numeric_cols_no_fecha = [col for col in numeric_cols if col != "Fecha"]
    selected_column = st.sidebar.selectbox(
        "Seleccione la columna a analizar:",
        numeric_cols_no_fecha
    )
    
    # Mostrar tabla raw
    st.header("Datos Raw")
    st.dataframe(df)
    
    # Ajustar modelos y encontrar el mejor
    results = fit_quantile_regression(df, selected_column, taus=[0.5])
    best_model = results.loc[results.groupby('columna')['pseudo_r2'].idxmax()]
    
    # Mostrar todos los modelos
    st.header("Comparación de Modelos")
    st.dataframe(results.sort_values('pseudo_r2', ascending=False))
    
    # Mostrar el mejor modelo
    #st.header("Mejor Modelo")
    #st.write(best_model)
    
    # Graficar el mejor ajuste
    fig = plot_best_fit(df, selected_column, best_model.iloc[0])
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.info("Asegúrese de que el archivo 'Seguimiento de datos.xlsx' está en la carpeta 'data/'") 
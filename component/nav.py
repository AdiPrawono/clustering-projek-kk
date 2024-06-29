import streamlit as st

def navbar():
    # Membuat 4 kolom
    col1, col2, col3, col4 = st.columns(4, gap="small")

    # Menambahkan konten di dalam kolom pertama
    with col1:
        st.page_link("HomePage.py", label="Home")
    with col2:
        st.page_link("pages/DataPreparation.py", label="Data Preparation")
    with col3:
        st.page_link("pages/preprocessing.py", label="Pre Processing")
    with col4:
        st.page_link("pages/modelling.py", label="Modelling")
    
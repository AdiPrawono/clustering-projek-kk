import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import pyswarm
from component.bootstrap import bootstrap
from component.nav import navbar

#Navbar
st.set_page_config(page_title="Kecerdasan komputasional", layout="wide")
navbar()

st.header('Metadata')
st.code("""
data = pd.read_csv("data-kk.csv", delimiter=",")
data
""")
data = pd.read_csv("data-kk.csv", delimiter=",")
st.write(data)

st.header("Menghapus kolom yang tidak relevan")
st.code("""
data.drop("NO", axis=1, inplace=True)
data.drop("NAMA", axis=1, inplace=True)
data.drop("NIK", axis=1, inplace=True)
data.drop("TGL LAHIR", axis=1, inplace=True)
""")
data.drop("NO", axis=1, inplace=True)
data.drop("NAMA", axis=1, inplace=True)
data.drop("NIK", axis=1, inplace=True)
data.drop("TGL LAHIR", axis=1, inplace=True)

st.write(data)

st.header("Metadata")
st.code("""
atribut = data.columns.tolist()
total_atribut = len(data.axes[1])
total_baris = len(data.axes[0])
""")

atribut = data.columns.tolist()
total_atribut = len(data.axes[1])
total_baris = len(data.axes[0])

st.write("Atribut: ",atribut)
st.write("Total Atribut: ", total_atribut)

st.header("Fitur Dari Dataset")
st.write("Fitur: ", data.dtypes)

st.header("Pengecekan Missing Value")
st.write("Missing Value", data.isnull().sum())

st.header("Prepocessing")
st.write("Usia : ", data['USIA'])
st.write("Kelamin : ", data['KELAMIN'])
st.write("Pendidikan : ", data['PENDIDIKAN'])
st.write("Pekerjaan : ", data['PEKERJAAN'])
st.write("Jumlah Penghasilan : ", data['JUMLAH PENGHASILAN'])

st.header("Data Mapping")
st.write("Fungsi Untuk Mapping Kolom Usia")
st.code("""
def map_age(age):
    if 22 <= age <= 35:
        return 0
    elif 36 <= age <= 45:
        return 1
    elif 46 <= age <= 55:
        return 2
    elif 56 <= age <= 75:
        return 3
    elif 76 <= age <= 86:
        return 4
data['USIA'] = data['USIA'].apply(map_age)
""")

st.write("Fungsi Untuk Mapping Gaji")
st.code("""
def map_gaji(gaji):
  proses = gaji.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "")
  gaji = int(proses)

  if gaji >= 2000000 and gaji <= 2500000:
    return 0
  elif gaji >= 2500001 and gaji <= 3000000:
    return 1
  elif gaji >= 3000001 and gaji <= 3500000:
    return 2
  elif gaji >= 3500001 and gaji <= 4000000:
    return 3
  elif gaji >= 4000001 and gaji <= 4500000:
    return 4
data['JUMLAH PENGHASILAN'] = data['JUMLAH PENGHASILAN'].apply(map_gaji)
""")
st.code("""
data['KELAMIN'] = data['KELAMIN'].map({'P': 0, 'L': 1})
data['PENDIDIKAN'] = data['PENDIDIKAN'].map({'TIDAK BERSEKOLAH': 0, 'SD': 1, 'SMP': 2, 'SMA': 3})
data['PEKERJAAN'] = data['PEKERJAAN'].map({'BURUH': 0, 'PETANI': 1, 'NELAYAN': 2, 'WIRAUSAHA': 3, 'PNS': 4})
""")

def map_gaji(gaji):
  proses = gaji.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "")
  gaji = int(proses)

  if gaji >= 2000000 and gaji <= 2500000:
    return 0
  elif gaji >= 2500001 and gaji <= 3000000:
    return 1
  elif gaji >= 3000001 and gaji <= 3500000:
    return 2
  elif gaji >= 3500001 and gaji <= 4000000:
    return 3
  elif gaji >= 4000001 and gaji <= 4500000:
    return 4
data['JUMLAH PENGHASILAN'] = data['JUMLAH PENGHASILAN'].apply(map_gaji)

data['KELAMIN'] = data['KELAMIN'].map({'P': 0, 'L': 1})
data['PENDIDIKAN'] = data['PENDIDIKAN'].map({'TIDAK BERSEKOLAH': 0, 'SD': 1, 'SMP': 2, 'SMA': 3})
data['PEKERJAAN'] = data['PEKERJAAN'].map({'BURUH': 0, 'PETANI': 1, 'NELAYAN': 2, 'WIRAUSAHA': 3, 'PNS': 4})

def map_age(age):
    if 22 <= age <= 35:
        return 0
    elif 36 <= age <= 45:
        return 1
    elif 46 <= age <= 55:
        return 2
    elif 56 <= age <= 75:
        return 3
    elif 76 <= age <= 86:
        return 4
data['USIA'] = data['USIA'].apply(map_age)

st.subheader("data setelah di mapping")
st.write("Usia : ", data['USIA'])
st.write("Kelamin : ", data['KELAMIN'])
st.write("Pendidikan : ", data['PENDIDIKAN'])
st.write("Pekerjaan : ", data['PEKERJAAN'])
st.write("Jumlah Penghasilan : ", data['JUMLAH PENGHASILAN'])




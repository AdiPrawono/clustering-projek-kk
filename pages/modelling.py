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


st.header("Data")
data = pd.read_csv("data-kk.csv", delimiter=",")

data.drop("NO", axis=1, inplace=True)
data.drop("NAMA", axis=1, inplace=True)
data.drop("NIK", axis=1, inplace=True)
data.drop("TGL LAHIR", axis=1, inplace=True)

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

st.write(data)




st.header("Proses Klasterasisasi")
st.header("Mencari Klaster Dengan Metode Elbow")
st.code("""
# Siapkan data untuk clustering
X = data[['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN']]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Metode Elbow untuk menentukan jumlah cluster optimal
sse = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=30)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot hasil Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_values, sse, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method For Optimal k')
plt.show()
""")
# Siapkan data untuk clustering
X = data[['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN']]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Metode Elbow untuk menentukan jumlah cluster optimal
sse = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=30)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot hasil Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_values, sse, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method For Optimal k')
st.pyplot(plt)

st.header("Score DBI dan Evaluasi Dari percobaan Jumlah klaster 2 hingga 10")
st.code("""
X = data[['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Davies-Bouldin Index analysis for determining the optimal number of clusters
dbi_scores = []
index = range(2, 11)

for i in index:
    kmeans = KMeans(n_clusters=i, random_state=40)
    labels = kmeans.fit_predict(X_scaled)
    dbi = davies_bouldin_score(X_scaled, labels)
    dbi_scores.append(dbi)
    print(f'Cluster {i}: DBI = {dbi}')

# Plot DBI vs. number of clusters
plt.figure(figsize=(10, 6))
plt.plot(index, dbi_scores, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index (DBI)')
plt.title('DBI vs. Number of Clusters')

""")
X = data[['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Davies-Bouldin Index analysis for determining the optimal number of clusters
dbi_scores = []
index = range(2, 11)

for i in index:
    kmeans = KMeans(n_clusters=i, random_state=40)
    labels = kmeans.fit_predict(X_scaled)
    dbi = davies_bouldin_score(X_scaled, labels)
    dbi_scores.append(dbi)
    st.write(f'Cluster {i}: DBI = {dbi}')

# Plot DBI vs. number of clusters
plt.figure(figsize=(10, 6))
plt.plot(index, dbi_scores, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index (DBI)')
plt.title('DBI vs. Number of Clusters')
st.pyplot(plt)

# Mengambil data dari dataset
X = data[['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
n_clusters = 5  # Jumlah cluster
kmeans = KMeans(n_clusters=n_clusters, random_state=30)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Dapatkan titik pusat (centroid) dari setiap cluster dalam skala asli
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Buat DataFrame untuk menampilkan centroid dalam bentuk yang lebih mudah dibaca
centroids_df = pd.DataFrame(centroids_original, columns=['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN'])

# Hitung Davies-Bouldin Index (DBI)
labels = kmeans.labels_
dbi = davies_bouldin_score(X_scaled, labels)
st.subheader('Score DBI')
st.write(f'Cluster : DBI = {dbi}')

# Mapping cluster labels to readable names
cluster_mapping = {0: "Klaster 1", 1: "Klaster 2", 2: "Klaster 3", 3: "Klaster 4", 4: "Klaster 5"}
data['Cluster'] = data['Cluster'].map(cluster_mapping)

# Menampilkan centroid dan data
st.subheader('Pusat Centroid')
st.write(centroids_df)
st.subheader('Data Setelah Di Klasterisasi')
st.write(data.head())
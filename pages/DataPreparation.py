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

st.set_page_config(page_title="Kecerdasan komputasional", layout="wide")
navbar()

st.header("Data Preparation")
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

st.header("Proses Klasterasisasi")
st.code("""
# mengambil data dari dataset
X = data[['USIA','KELAMIN','PENDIDIKAN','PEKERJAAN','JUMLAH PENGHASILAN']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=30)
data['Cluster'] = kmeans.fit_predict(X_scaled)
# Dapatkan titik pusat (centroid) dari setiap cluster dalam skala asli
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Buat DataFrame untuk menampilkan centroid dalam bentuk yang lebih mudah dibaca
centroids_df = pd.DataFrame(centroids_original, columns=['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN'])


cluster_mapping = {0: "Klaster 1", 1: "Klaster 2", 2: "Klaster 3", 3: "Klaster 4",4: "Klaster 5"}
data['Cluster'] = data['Cluster'].map(cluster_mapping)
centroids_df
data.head()
        

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='twilight', s=100)
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
""")
# mengambil data dari dataset
X = data[['USIA','KELAMIN','PENDIDIKAN','PEKERJAAN','JUMLAH PENGHASILAN']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=30)
data['Cluster'] = kmeans.fit_predict(X_scaled)
# Dapatkan titik pusat (centroid) dari setiap cluster dalam skala asli
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Buat DataFrame untuk menampilkan centroid dalam bentuk yang lebih mudah dibaca
centroids_df = pd.DataFrame(centroids_original, columns=['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN'])


cluster_mapping = {0: "Klaster 1", 1: "Klaster 2", 2: "Klaster 3", 3: "Klaster 4",4: "Klaster 5"}
data['Cluster'] = data['Cluster'].map(cluster_mapping)
st.write("Nilai Centroid :", centroids_df)
st.write("Hasil Klasteriasasi : ",data.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='twilight', s=100)
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot(plt)

st.header("Optimasi Centroid")
st.code("""
X = data[['USIA','KELAMIN','PENDIDIKAN','PEKERJAAN','JUMLAH PENGHASILAN']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster
n_clusters = 5

# Fungsi objektif untuk PSO (menggunakan Davies-Bouldin Index)
def objective_function(centroids_flat):
    centroids = centroids_flat.reshape(n_clusters, -1)
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    labels = kmeans.fit_predict(X_scaled)
    score = davies_bouldin_score(X_scaled, labels)
    return score

# Batasan untuk PSO
bounds = [(X_scaled.min(axis=0)[i], X_scaled.max(axis=0)[i]) for i in range(X_scaled.shape[1])] * n_clusters
lb, ub = np.array(bounds).T

# Optimasi menggunakan PSO
optimal_centroids, fopt = pyswarm.pso(objective_function, lb, ub, swarmsize=100, maxiter=200)

# Bentuk ulang hasil optimal
optimal_centroids = optimal_centroids.reshape(n_clusters, -1)

# Terapkan K-Means dengan centroid optimal
kmeans_pso = KMeans(n_clusters=n_clusters, init=optimal_centroids, n_init=1)
data['Cluster'] = kmeans_pso.fit_predict(X_scaled)

# Dapatkan titik pusat (centroid) dari setiap cluster dalam skala asli
centroids_original = scaler.inverse_transform(kmeans_pso.cluster_centers_)

# Buat DataFrame untuk menampilkan centroid dalam bentuk yang lebih mudah dibaca
centroids_df = pd.DataFrame(centroids_original, columns=['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN'])

# Davies-Bouldin Index analysis for determining the optimal number of clusters


kmeans = KMeans(n_clusters=5, random_state=30)
labels = kmeans.fit_predict(X_scaled)
dbi = davies_bouldin_score(X_scaled, labels)
# dbi_scores.append(dbi)
print(f'Cluster : DBI = {dbi}')

# Tampilkan hasil
print("Titik Pusat (Centroid) dari setiap Cluster dalam skala asli:")
print(centroids_df)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='coolwarm', s=100)
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
""")
X = data[['USIA','KELAMIN','PENDIDIKAN','PEKERJAAN','JUMLAH PENGHASILAN']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster
n_clusters = 5

# Fungsi objektif untuk PSO (menggunakan Davies-Bouldin Index)
def objective_function(centroids_flat):
    centroids = centroids_flat.reshape(n_clusters, -1)
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    labels = kmeans.fit_predict(X_scaled)
    score = davies_bouldin_score(X_scaled, labels)
    return score

# Batasan untuk PSO
bounds = [(X_scaled.min(axis=0)[i], X_scaled.max(axis=0)[i]) for i in range(X_scaled.shape[1])] * n_clusters
lb, ub = np.array(bounds).T

# Optimasi menggunakan PSO
optimal_centroids, fopt = pyswarm.pso(objective_function, lb, ub, swarmsize=100, maxiter=200)

# Bentuk ulang hasil optimal
optimal_centroids = optimal_centroids.reshape(n_clusters, -1)

# Terapkan K-Means dengan centroid optimal
kmeans_pso = KMeans(n_clusters=n_clusters, init=optimal_centroids, n_init=1)
data['Cluster'] = kmeans_pso.fit_predict(X_scaled)

# Dapatkan titik pusat (centroid) dari setiap cluster dalam skala asli
centroids_original = scaler.inverse_transform(kmeans_pso.cluster_centers_)

# Buat DataFrame untuk menampilkan centroid dalam bentuk yang lebih mudah dibaca
centroids_df = pd.DataFrame(centroids_original, columns=['USIA', 'KELAMIN', 'PENDIDIKAN', 'PEKERJAAN', 'JUMLAH PENGHASILAN'])

# Davies-Bouldin Index analysis for determining the optimal number of clusters


kmeans = KMeans(n_clusters=5, random_state=30)
labels = kmeans.fit_predict(X_scaled)
dbi = davies_bouldin_score(X_scaled, labels)
# dbi_scores.append(dbi)
st.write(f'Cluster : DBI = {dbi}')

# Tampilkan hasil
st.write("Titik Pusat (Centroid) dari setiap Cluster dalam skala asli:")
st.write(centroids_df)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='coolwarm', s=100)
plt.title('K-Means Clustering with PSO')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot(plt)
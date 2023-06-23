import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fungsi untuk mengubah jenis kelamin menjadi angka biner (0 atau 1)
def encode_gender(gender):
    if gender == 'L':
        return 1
    else:
        return 0

# Membaca data pemilu dari file CSV
data = pd.read_csv('pemilu2.csv')

# Mengubah kolom jenis kelamin menjadi angka biner
data['Jenis Kelamin'] = data['Jenis Kelamin'].apply(encode_gender)

# Mengambil kolom yang akan digunakan dalam clustering
features = data[['Usia', 'Jenis Kelamin', 'RT', 'RW']]

# Membuat tampilan aplikasi menggunakan Streamlit
st.title('Aplikasi Klastering Data Mining Pemilu')

# Menampilkan data asli
st.subheader('Data Asli')
st.text('Laki-Laki = 1')
st.text('Perempuan = 0')
st.write(data)

# Menambahkan slider untuk memilih jumlah klaster
num_clusters = st.slider('Jumlah Klaster', min_value=2, max_value=10, value=3, step=1)

# Menggunakan algoritma K-means untuk membuat klaster
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(features)

# Menambahkan kolom klaster ke dalam data
data['Cluster'] = kmeans.labels_

# Menampilkan data yang sudah dikelompokkan ke dalam klaster
st.subheader('Informasi Klaster')
st.write(data[['Nama', 'Cluster']])

# Menampilkan informasi jumlah data dalam setiap klaster
st.subheader('Jumlah klaster')
st.write(data['Cluster'].value_counts())

# Menghilangkan kolom 'Nama' dan 'Jenis Kelamin' karena tidak relevan untuk klastering
data = data.drop(['Nama', 'Jenis Kelamin'], axis=1)

# Memilih kolom yang akan digunakan untuk klastering
features = data[['Usia', 'RT', 'RW']]

# Menambahkan kolom 'Klaster' ke dalam data
data['Klaster'] = kmeans.labels_
st.text(f'Jumlah data = {len(data)}')

# Menentukan skema warna untuk centroid klaster
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'pink']
color_map = dict(zip(range(num_clusters), colors))

# Menampilkan scatter plot klastering
plt.scatter(features['RW'], features['Usia'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('RW')
plt.ylabel('Usia')

# Menampilkan centroid klaster dengan warna yang berbeda
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[2], centroid[0], marker='x', color=color_map[i], label=f'Centroid {i}')
plt.legend()

st.pyplot(plt)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv(r"C:\Users\javie\OneDrive\Desktop\Excel_DB\customer_data.csv")
print(data.head())

print(data.info())

print(data.describe())

# Normalizar los datos 
escalador = MinMaxScaler()
data_escalada = escalador.fit_transform(data[['Edad','Ingresos Anuales (k$)', 'Puntuación de Gasto (1-100)']])
print(data_escalada)

# Aplicar PCA
pca = PCA(n_components=2, random_state=42)
pca_resultados = pca.fit_transform(data_escalada)
# Aplicar SVD
U, Sigma, VT = np.linalg.svd(data_escalada)
# Seleccionar los dos primeros componentes singulares para reducción de dimensiones
k = 2
svd_resultados = U[:, :k] * Sigma[:k]

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(data_escalada)

# Aplicar Clustering Jerárquico 
linked = linkage(data_escalada, method='ward')


# VISUALIZACIÓN DE DATOS 

#Crear un gráfico de dispersión para los resultados de PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_resultados[:, 0], y=pca_resultados[:, 1], hue=kmeans_clusters, palette='viridis', s=100)
plt.title('Gráfico de dispersión de los resultados de PCA')
plt.xlabel('Componentes Principal 1')
plt.ylabel('Componentes Principal 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Crear un dendrogramma del clústering jerárquico
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrograma de Clústering Jerárquico')
plt.xlabel('Indice de Muestra')
plt.ylabel('Distancia Ward')
plt.axhline(y=10, color='r', linestyle='--')
plt.show()

# 5. INTERPRETACIÓN Y ESTRATEGIAS DE MARKETING BASADAS EN LOS CLUSTERS
# su edad media
#su ingresos medios
# su puntuación de gastos media

# Análisis de clusters para determinar estrategias
cluster_info = pd.DataFrame({
    'Cluster': kmeans_clusters,
    'Edad': data['Edad'],
    'Ingresos': data['Ingresos Anuales (k$)'],
    'Gasto': data['Puntuación de Gasto (1-100)']
})

# Descripción de cada clúster
for cluster in cluster_info['Cluster'].unique():
    cluster_data = cluster_info[cluster_info['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(f" - Edad Media: {cluster_data['Edad'].mean(): .0f}")
    print(f" - Ingresos Medios: {cluster_data['Ingresos'].mean(): .2f}")
    print(f" - Puntuación de Gasto Media: {cluster_data['Gasto'].mean(): .2f}")
    print()

    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

np.random.seed(42)
data = np.random.rand(100, 2)
print(data)

kmeans1 = KMeans(n_clusters=5, random_state=42)
kmeans1.fit(data)
centroides1 = kmeans1.cluster_centers_
print(centroides1)

etiquetas1 = kmeans1.labels_
print(etiquetas1)

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=etiquetas1, cmap='viridis', marker='o')
plt.scatter(centroides1[:, 0], centroides1[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.title('Visualización de K-Means Clustering')
plt.legend()
plt.show()

np.random.seed(4)
data1 = np.random.rand(100, 2)

pinguinos = sns.load_dataset('penguins')
pinguinos.dropna(inplace=True)
data = pinguinos[['bill_length_mm', 'bill_depth_mm']]

kmeans2 = KMeans(n_clusters=3, random_state=42)
kmeans2.fit(data)

centroides2 = kmeans2.cluster_centers_
etiquetas2 = kmeans2.labels_

plt.figure(figsize=(12, 7))
sns.scatterplot(data=pinguinos, x='bill_length_mm', y='bill_depth_mm', hue=etiquetas2, palette='viridis')
plt.scatter(centroides2[:, 0], centroides2[:, 1], c='red', s=100, marker='x', label='Centroides')
plt.title('K-Means Clustering en el conjunto de datos Pinguinos')
plt.legend()    
plt.xlabel('Longitud del Pico (mm)')
plt.ylabel('Profundidad del Pico (mm)')
plt.show()
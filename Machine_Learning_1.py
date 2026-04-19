from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#Clustering Jerarquico

iris = load_iris()
X = iris.data
linked = linkage(X, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, 
            orientation='top',
            labels=iris.target,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Dendograma de Clústering")
plt.show()

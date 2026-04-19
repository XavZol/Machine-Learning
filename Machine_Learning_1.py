import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Gráfico de disperción lineal

iris = load_iris()
X = iris.data
y = iris.target

print(y)

X_centrado = X - np.mean(X, axis=0)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_centrado)

especies = ["setosa", "versicolor", "virginica"]
plt.figure(figsize=(8, 6))
for i in range(0, 3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=especies[i])
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.legend()
plt.title('PCA del conjunto Iris')  
plt.show()
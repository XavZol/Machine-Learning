import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# Singular Value Descomposition (SVD)

A = np.array([[1, 2],
            [3, 4], 
            [5, 6]])

U, sigma, VT = np.linalg.svd(A)

print("U =",U)
print("sigma =",sigma)
print("VT =", VT)

iris = load_iris()
X = iris.data
X_centrado = X - np.mean(X, axis=0)

# Matrices 
U, sigma, VT = np.linalg.svd(X_centrado)

# Reducir la Dimensionalidad de un conjunto de datos
k = 2
X_transformado = U[:, :k] * sigma[:k]

especies = ["setosa", "versicolor", "virginica"]

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_transformado[iris.target == i, 0], 
                X_transformado[iris.target == i, 1], 
                label=especies[i])
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.legend()
    plt.title('Dataset iris transformado por SVD');
plt.show()
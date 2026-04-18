import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns

df = sns.load_dataset("iris")
print(df.head())

sp = df.species.unique()
print(sp)

X = df.drop("species", axis=1)
print(X)

le = LabelEncoder()
especies= le.fit_transform(df["species"])
print(especies)

y = especies
print(y)

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, train_size=0.8, random_state=42)
arbol = DecisionTreeClassifier()
arbol.fit(X_entrena, y_entrena)
plt.figure(figsize=(10,10))
plot_tree(decision_tree=arbol, class_names=["setosa", "versicolor", "virginica"], feature_names=df.columns.to_list(), filled=True)
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ruta = r"C:\Users\javie\OneDrive\Desktop\Excel_DB\tarjetas_credito.csv"

df = pd.read_csv(ruta)
print(df.head())

escala = MinMaxScaler(feature_range=(0, 1))
normado = escala.fit_transform(df)
df_normado = pd.DataFrame(data=normado, columns=df.columns)
print(df_normado.head())

# VAriables Dependientes e Independientes
X = df_normado.drop("Clase", axis=1)
y = df_normado["Clase"]
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, train_size=0.7, random_state=42)

modelos = [
    ("Regresion Logística", LogisticRegression()),
    ("Arbol de Decisión", DecisionTreeClassifier()),
    ("Bosque Aleatorio", RandomForestClassifier())
]
for nombre, modelo in modelos:
    modelo.fit(X_entrena, y_entrena)
    puntaje = modelo.score(X_prueba,y_prueba)
    print(f"{nombre}: {puntaje: .4f}")
    
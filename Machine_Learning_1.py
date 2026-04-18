import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

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
forest = RandomForestClassifier()
forest.fit(X_entrena, y_entrena)
forest.score(X_prueba, y_prueba)

nuevo_registro = pd.DataFrame({
    'Duracion': [0.000006], 'V1': [0.452345], 'V2': [0.564789], 'V3': [0.123456], 'V4': [0.654321],
    'V5': [0.987654], 'V6': [0.345678], 'V7': [0.234567], 'V8': [0.876543], 'V9': [0.456789],
    'V10': [0.567890], 'V11': [0.678901], 'V12': [0.789012], 'V13': [0.890123], 'V14': [0.901234],
    'V15': [0.012345], 'V16': [0.543210], 'V17': [0.432109], 'V18': [0.321098], 'V19': [0.210987],
    'V20': [0.109876], 'V21': [0.098765], 'V22': [0.887654], 'V23': [0.776543], 'V24': [0.665432],
    'V25': [0.554321],     'V26': [0.443210], 'V27': [0.332109], 'V28': [0.221098], 'Monto': [0.110987]
}, index=[0])

clase_predicha = forest.predict(nuevo_registro)
print(clase_predicha)

probabilidades = forest.predict_proba(nuevo_registro)

print("Clase predicha: ", clase_predicha[0])
print("Probabilidades de legitimidad: ", probabilidades[0][0])
print("Probabilidades de fraude: ", probabilidades[0][1])


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ruta = r"C:\Users\javie\OneDrive\Desktop\Excel_DB\datos_seguro.csv"
df = pd.read_csv(ruta)
print(df.head())

plt.scatter(df.edad, 
            df.compra)
plt.show()

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(df[["edad"]], df["compra"], train_size=0.9) 
print(X_entrena)
print(X_prueba)

modelo = LogisticRegression()
modelo.fit(X_entrena, y_entrena)
print(modelo.score(X_entrena, y_entrena))

datos_nuevos = pd.DataFrame({"edad":[25, 35, 45, 55]})
print(datos_nuevos)

probabilidades = modelo.predict_proba(datos_nuevos)
print(probabilidades)

prob_compra = probabilidades[:, 1]
print(prob_compra)

plt.scatter(df.edad, 
            df.compra)
plt.scatter(datos_nuevos["edad"],
            prob_compra,
            color="red")
plt.show()
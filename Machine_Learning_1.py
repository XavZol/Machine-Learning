import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ruta = r"C:\Users\javie\OneDrive\Desktop\Excel_DB\Ventas 3.csv"

df = pd.read_csv(ruta)
print(df.head())

df.info()

df["Fecha"] = pd.to_datetime(df['Fecha'])
print(df.head())

escala = MinMaxScaler(feature_range=(0, 1))
columnas_escalar = df.drop(["Ventas", "Fecha"], axis=1).columns
normado = escala.fit_transform(df[columnas_escalar])
df_normado = pd.DataFrame(data=normado, columns=columnas_escalar)

df_normado["Ventas"] = df["Ventas"]
df_normado["Fecha"] = df["Fecha"]

print(df_normado.head())
df.info()

# Identificar las variables pendientes e independientes
X = df_normado.drop(['Ventas', 'Fecha'], axis=1)
y = df["Ventas"]

# Dividir en conjuntos de entrenamiento y pruebas
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y , train_size=0.8, random_state=42)
print(df.describe())

# Visualización de la distribución de ventas
plt.figure(figsize=(12, 6))
sns.histplot(df['Ventas'], bins=30, kde=True)
plt.title('Distribución de ventas')
plt.xlabel('Ventas')
plt.ylabel('Frecuencia')
plt.show()

# Relación entre ventas y día  de la semana con promedio de ventas
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='DíaDeLaSemana', y='Ventas', color="red")
plt.title('Ventas promedio por Día de la Semana')
plt.xlabel('Día de la Semana')
plt.ylabel('Ventas Promedio')
plt.show()

# boxplot para comparar las ventas con y sin promociones.
plt.figure(figsize=(12, 6))
sns.boxplot(x='Promociones', y='Ventas', data=df)
plt.title('Efecto de las Promociones en las Ventas')

# boxplot para comparar las ventas para los días normales y festivos
plt.figure(figsize=(12,6))
sns.boxplot(x='Festivo', y='Ventas', data=df)
plt.title('Efecto de los Días Festivos en las Ventas')

# boxplot para ver la interacción entre promociones y días festivos en las ventas.
plt.figure(figsize=(12,6))
sns.boxplot(x='Promociones', y='Ventas', hue='Festivo', data=df)
plt.title('Interacción entre Promociones y Días Festivos en las Ventas')
plt.show()

# almacenar modelos
modelos = [
    ("modelo lineal", LinearRegression()),
    ("modelo arbol", DecisionTreeRegressor(random_state=42)),
    ("modelo bloque", RandomForestRegressor(random_state=42))
]

# Entrenar modelos y mostrar puntajes
for nombre, modelo in modelos:
    modelo.fit(X_entrena, y_entrena)
    puntaje = modelo.score(X_prueba, y_prueba)
    print(f'{nombre}: {puntaje}')

# Alojamos el modelo de Regresión Líneal en una variable
modelo_lineal = LinearRegression()

#Entrenamos el modelo con los datos de entrenamiento
modelo_lineal.fit(X_entrena, y_entrena)

# Realizamos predicciones usando el conjunto de prueba
predicciones_lineal = modelo_lineal.predict(X_prueba)

plt.figure(figsize=(10, 6))
plt.scatter(y_prueba, predicciones_lineal, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Ventas Reales vs Ventas Predichas')
plt.show()


df_test = pd.DataFrame({'Real': y_prueba, 'Predicho': predicciones_lineal})
df_test = df_test.sort_index()

plt.figure(figsize=(15, 5))
plt.plot(df_test['Real'], label='Ventas Reales', alpha=0.7)
plt.plot(df_test['Predicho'], label='Ventas Predichas', alpha=0.7)
plt.legend()
plt.title('Comparación de Ventas Reales y Ventas Predichas a lo largo del tiempo')
plt.show()

# 1 EI primer gráfico. que muestra un diagrama de dispersión de las Ventas Reales vs Ventas Predichas.
#  sugiere que el modelo de regresión lineal está haciendo un buen trabajo al predecir las ventas, 
# La linea de tendencia indica una fuerte relación posmva entre los valores reales y predichos, 
# 10 que es un signo prometedor de que el modelo puede capturar la tendencia de las ventas con eficacia 2 EI 
# segundo gráfico compara las Ventas Reales y las Ventas a 10 largo del tiempo y también parece seguir un patrón similar,
#  aunque hay algunos puntos en los que las predicciones y los valores reales difieren significativamente. Esto puede deberse a
#  eventos no capturados por las variables en tu modelo o a variaciones naturales en las ventas que no son predecibles Aquí hay
#  algunalrecomendaciones para la tienda minorista Optimización de Inventario. Utiliza las predcciones para gestionar mejor el 
# inventario Las fechas festivas pueden requerir un stock adicional para evitar la fata de productos. Planificación de Personal: 
# Austa los horarios del personal según días festivos y no necesariamente según dias de promociones Marketing Dirigido. 
# Si identificas patrones de cuándo las ventas son más fuertes, puedes dirigir las campañas de marketing para esos periodos y 
# potencialmente aumentar aún más las ventas Análisis de Anomalías Investiga aquellos puntos donde hay grandes desviaciones entre 
# las ventas reales y las predichas para entender mejor los factores no capturados por el modelo Mejoras en el Modelo: Considera 
# incluir más variables en el modelo que puedan afectar las ventas, como datos económicos generales,
#  eventos locales, competencia, o incluso el clima
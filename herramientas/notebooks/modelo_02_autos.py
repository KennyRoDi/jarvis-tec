"""Genera backend/features/modelo_02_autos/notebook.ipynb (ver nb.py)."""
from nb import Cuaderno

c = Cuaderno()
c.portada(
    problema="""Queremos estimar el precio de reventa de un automóvil usado a partir de sus características: año, precio de agencia actual, kilometraje, combustible, tipo de vendedor, transmisión y número de dueños previos.

Es un problema de **regresión supervisada**. La variable objetivo es `Selling_Price` y está expresada en *lakhs* de rupias indias (1 lakh = 100 000 INR).""",
    titulo="Caso Aplicado: Precio de Reventa de un Automóvil",
    subtitulo="Regresión lineal y Random Forest con variables numéricas y categóricas",
    temas="Variables categóricas · Pipeline · Regresión lineal · Random Forest · Validación cruzada",
    herramientas="Python · Pandas · Scikit-learn · Seaborn",
    hilo=[("1 — Entendimiento", "¿Qué forma tienen los datos y hay valores faltantes?"),
          ("2 — Exploración", "¿Qué variables se relacionan con el precio de venta?"),
          ("3 — Modelo", "¿Un Random Forest mejora a una regresión lineal con el mismo preprocesamiento?"),
          ("4 — Predicción", "¿Cuánto se espera que valga un auto con ciertas características?")],
    nota="este es el modelo de referencia del proyecto: todo el preprocesamiento va dentro de un Pipeline, de modo que se use exactamente igual al entrenar y al predecir.",
    introduccion="Este notebook es un extra del modelo 02 del proyecto: reproduce de forma didáctica el entrenamiento que hace `train.py` y se puede probar en Google Colab. No reemplaza al código del repositorio.")

c.md("### Librerías")
c.code("""# Manejo de datasets
import pandas as pd
import numpy as np
# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Modelos
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Ignorar Warnings
import warnings
warnings.filterwarnings('ignore')

SEMILLA = 42""")

c.md("""---
# Sección 1: Entendimiento de los Datos
---

## Entendimiento de los Datos

El conjunto tiene 301 registros y 9 columnas:

- Car_Name      : variable cualitativa nominal, nombre del modelo (se descarta, se explica abajo)
- Year          : variable cuantitativa, año de fabricación
- Selling_Price : variable objetivo, precio de venta en lakhs
- Present_Price : variable cuantitativa, precio de agencia actual en lakhs
- Kms_Driven    : variable cuantitativa, kilometraje recorrido
- Fuel_Type     : variable cualitativa nominal (Petrol, Diesel, CNG)
- Seller_Type   : variable cualitativa nominal (Dealer, Individual)
- Transmission  : variable cualitativa nominal (Manual, Automatic)
- Owner         : variable cuantitativa, dueños anteriores""")
c.datos("modelo_02_autos", "Son 301 filas y 9 columnas.")
c.code("""# Cargar los datos
datos = pd.read_csv(RUTA)
datos.head()""")
c.code("""# Describir el dataset
datos.describe()""")
c.code("""# Información del dataset
datos.info()""")
c.code("""print("Cantidad de registros:", len(datos))
print("Cantidad de columnas:", len(datos.columns))
print("Filas y columnas:", datos.shape)
print("Valores nulos:", int(datos.isnull().sum().sum()))
print("Nombres de auto distintos:", datos["Car_Name"].nunique(), "en", len(datos), "filas")""")
c.md("""`Car_Name` tiene cerca de 100 valores distintos en 301 filas. Con tan pocos ejemplos por nombre no aporta capacidad de generalización, así que se descarta.""")
c.code("""df = datos.copy()
df.columns = [c.lower() for c in df.columns]
df = df.drop(columns=["car_name"])
df.head()""")

c.md("""---
# Sección 2: Exploración de los Datos
---

## Exploración de los Datos""")
c.code("""OBJETIVO = "selling_price"
NUMERICAS = ["year", "present_price", "kms_driven", "owner"]
CATEGORICAS = ["fuel_type", "seller_type", "transmission"]

sns.histplot(df[OBJETIVO], kde=True)
plt.title("Distribución del precio de venta")
plt.show()
print("Asimetría: %.2f   Media: %.2f   Máximo: %.2f" % (df[OBJETIVO].skew(), df[OBJETIVO].mean(), df[OBJETIVO].max()))""")
c.md("La distribución tiene una asimetría positiva marcada: la mayoría de los autos cuesta poco y unos pocos llegan a precios muy altos.")
c.code("""sns.heatmap(df[NUMERICAS + [OBJETIVO]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre variables numéricas")
plt.show()""")
c.md("""La variable con mayor correlación con el precio de venta es `present_price`, el precio de agencia actual.

> Analogía: el precio de agencia es como el precio de lista de un auto nuevo. Un auto usado vale una fracción de ese precio, y esa fracción depende de la edad, el kilometraje y el estado.""")
c.code("""sns.scatterplot(data=df, x="present_price", y=OBJETIVO, hue="fuel_type")
plt.title("Precio actual vs. precio de venta")
plt.show()
df.groupby("fuel_type")[[OBJETIVO, "present_price"]].mean().round(2)""")
c.md("La relación es aproximadamente lineal. Los autos diésel tienen un precio de agencia y un precio de venta promedio más altos que los de gasolina.")

c.md("""---
# Sección 3: Ajuste del Modelo
---

## Modelo de Machine Learning

Se separa el 20 % de los datos para la prueba.""")
c.code("""X, y = df[NUMERICAS + CATEGORICAS], df[OBJETIVO]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
print("Entrenamiento:", X_train.shape, " Prueba:", X_test.shape)""")
c.md("> Es importante siempre validar los rangos de los conjuntos creados, para evitar caer en extrapolación:")
c.code("""X_train[NUMERICAS].describe().loc[["min", "max"]].join(X_test[NUMERICAS].describe().loc[["min", "max"]], rsuffix="_prueba")""")
c.md("""### Pipeline

El preprocesamiento (estandarizar las variables numéricas y codificar las categóricas con one-hot) va dentro del mismo objeto que el estimador. Así, lo que se aplica al entrenar es exactamente lo que se aplica al predecir un auto nuevo.

La **línea base** es una regresión lineal con el mismo preprocesamiento. El modelo principal es un **Random Forest** de 200 árboles.""")
c.code("""def construir_modelo(estimador):
    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    return Pipeline([("preprocesador", preprocesador), ("modelo", estimador)])

base = construir_modelo(LinearRegression()).fit(X_train, y_train)
modelo = construir_modelo(RandomForestRegressor(n_estimators=200, random_state=SEMILLA))

# Validación cruzada de 5 particiones, solo con el entrenamiento
cv_r2 = cross_val_score(modelo, X_train, y_train, cv=5, scoring="r2")
print("R2 en validación cruzada (Random Forest): %.3f ± %.3f" % (cv_r2.mean(), cv_r2.std()))
modelo.fit(X_train, y_train)""")

c.md("##### Evaluación en el conjunto de prueba")
c.code("""def metricas(real, pred):
    return {"R2": r2_score(real, pred), "MAE": mean_absolute_error(real, pred), "RMSE": np.sqrt(mean_squared_error(real, pred))}

y_pred = modelo.predict(X_test)
pd.DataFrame({"Regresión lineal (línea base)": metricas(y_test, base.predict(X_test)),
              "Random Forest": metricas(y_test, y_pred)}).T.round(3)""")
c.code("""plt.scatter(y_test, y_pred, alpha=0.7)
limite = max(y_test.max(), y_pred.max())
plt.plot([0, limite], [0, limite], "r--")
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Real vs. predicho (prueba)")
plt.show()""")
c.md("""El Random Forest reduce el error de la línea base casi a la mitad. El R2 de prueba es mayor que el de validación cruzada: con solo 61 filas de prueba y precios muy asimétricos, la prueba es una estimación ruidosa y conviene tomar la validación cruzada como referencia más prudente.""")

c.md("### Predicción para un auto nuevo")
c.code("""auto = pd.DataFrame([{"year": 2016, "present_price": 8.5, "kms_driven": 30000, "owner": 0,
                      "fuel_type": "Petrol", "seller_type": "Dealer", "transmission": "Manual"}])
print("Precio de venta estimado: %.2f lakhs" % modelo.predict(auto[NUMERICAS + CATEGORICAS])[0])""")

c.md("""---
# Conclusiones
---

## Resultados

Con las características básicas de un auto se puede estimar su precio de reventa con un error medio de aproximadamente 0.6 lakhs. El Random Forest mejora a la regresión lineal: el R2 de prueba pasa de 0.85 a 0.96.

Limitaciones: el conjunto tiene solo 301 registros y viene del mercado indio, así que no se debe usar para otros mercados. El R2 de validación cruzada (0.88) es menor que el de prueba, por el tamaño reducido de ambos conjuntos. Los precios muy altos son pocos (el máximo es 35 lakhs frente a una mediana de 3.6).""")

if __name__ == "__main__":
    print(c.guardar("modelo_02_autos"))

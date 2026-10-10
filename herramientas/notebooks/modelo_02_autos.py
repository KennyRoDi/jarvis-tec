"""Genera backend/features/modelo_02_autos/notebook.ipynb (ver nb.py)."""
from nb import Cuaderno

c = Cuaderno()
c.portada(
    problema="""Queremos estimar el precio de reventa de un automóvil usado a partir de siete datos que su dueño conoce: año, precio de agencia actual, kilometraje, combustible, tipo de vendedor, transmisión y número de dueños anteriores.

Es un problema de **regresión supervisada**. La variable objetivo es `Selling_Price`, en *lakhs* de rupias indias (1 lakh = 100 000 INR). Hay solo unos 300 autos y unos pocos son muy caros, lo que condiciona la elección del modelo.""",
    titulo="Caso Aplicado: Precio de Reventa de un Automóvil",
    subtitulo="Regresión con pocos datos: modelar la razón de reventa y no el precio",
    temas="Variables categóricas · Pipeline · Validación cruzada repetida · Extrapolación de los árboles · Cambio del objetivo",
    herramientas="Python · Pandas · Scikit-learn · Seaborn",
    hilo=[("1 — Entendimiento y limpieza", "¿Hay filas repetidas, categorías con casi ningún ejemplo y una regularidad que se pueda aprovechar?"),
          ("2 — Exploración", "¿Qué variables se relacionan con el precio y con la razón de reventa?"),
          ("3 — Modelo", "¿Mejora modelar la razón reventa / agencia frente a modelar el precio?"),
          ("4 — Evaluación", "¿Qué pasa con autos más caros que cualquiera del entrenamiento?")],
    nota="con pocos datos, un buen R2 de prueba puede ser un accidente de la partición. Por eso los candidatos se comparan con validación cruzada repetida, y las cifras de la prueba se leen con sus intervalos.",
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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
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
c.code("""# Información del dataset
datos.info()""")
c.code("""print("Cantidad de registros:", len(datos))
print("Valores nulos:", int(datos.isnull().sum().sum()))
print("Filas duplicadas:", int(datos.duplicated().sum()))
print("Nombres de auto distintos:", datos["Car_Name"].nunique(), "en", len(datos), "filas")""")
c.md("""`Car_Name` tiene cerca de 100 valores distintos en 301 filas: con tan pocos ejemplos por nombre no generaliza, así que se descarta. Hay **2 filas duplicadas**. Se quitan **antes** de dividir, para que una copia no caiga en el entrenamiento y la otra en la prueba.""")
c.code("""df = datos.drop(columns=["Car_Name"]).drop_duplicates().reset_index(drop=True)
df.columns = [c.lower() for c in df.columns]
print("Quedan", len(df), "autos")
print("Combustible:", df["fuel_type"].value_counts().to_dict())
print("Dueños anteriores:", df["owner"].value_counts().sort_index().to_dict())
df.head()""")
c.md("""Hay categorías casi sin ejemplos: **2 autos de gas natural (CNG)** y **un solo auto con 3 dueños**. El modelo no puede aprender nada fiable de ellos, y la API del proyecto avisa cuando se pregunta por esos casos.

### Una regularidad que se puede aprovechar

Veamos la razón entre el precio de reventa y el de agencia.""")
c.code("""razon = df["selling_price"] / df["present_price"]
print("Razón reventa / agencia: mínimo %.2f, mediana %.2f, máximo %.2f" % (razon.min(), razon.median(), razon.max()))
print("Razón mediana por año:", razon.groupby(df["year"]).median().round(2).to_dict())""")
c.md("La razón **nunca supera 1** y sube de forma regular con el año: un auto de 2003 vale alrededor de una quinta parte de su precio de agencia, y uno de 2017 casi el 90 %. Más adelante se usa esto para elegir qué predecir.")

c.md("""---
# Sección 2: Exploración de los Datos
---

## Exploración de los Datos""")
c.code("""OBJETIVO = "selling_price"
NUMERICAS = ["year", "present_price", "kms_driven", "owner"]
CATEGORICAS = ["fuel_type", "seller_type", "transmission"]
VARIABLES = NUMERICAS + CATEGORICAS

sns.histplot(df[OBJETIVO], kde=True)
plt.title("Distribución del precio de venta")
plt.show()
print("Asimetría: %.2f   Media: %.2f   Mediana: %.2f   Máximo: %.2f" % (df[OBJETIVO].skew(), df[OBJETIVO].mean(), df[OBJETIVO].median(), df[OBJETIVO].max()))""")
c.md("La distribución tiene una asimetría positiva marcada: la mayoría de los autos cuesta poco y unos pocos llegan a precios muy altos.")
c.code("""sns.heatmap(df[NUMERICAS + [OBJETIVO]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre variables numéricas")
plt.show()
print("Correlación del kilometraje con el año: %.2f" % df["kms_driven"].corr(df["year"]))""")
c.md("""El precio de agencia es, con mucha ventaja, la variable más asociada al precio de reventa. El kilometraje casi no se asocia por sí solo, porque va mezclado con la antigüedad.

> Analogía: el precio de agencia es como el precio de lista de un auto nuevo. Un auto usado vale una fracción de ese precio, y esa fracción depende sobre todo de la edad.""")
c.code("""sns.scatterplot(data=df, x="present_price", y=OBJETIVO, hue="fuel_type")
plt.title("Precio actual vs. precio de venta")
plt.show()
df.groupby("fuel_type")[[OBJETIVO, "present_price"]].median().round(2)""")
c.md("La relación es aproximadamente lineal, con más dispersión en los autos caros. Los diésel tienen una reventa mediana mayor, pero también un precio de agencia mayor.")
c.code("""razones = razon.groupby(df["year"])
plt.boxplot([g.to_numpy() for _, g in razones], tick_labels=[str(a) for a, _ in razones])
plt.xticks(rotation=45)
plt.ylabel("Reventa / precio de agencia")
plt.title("La razón de reventa baja con la antigüedad del auto")
plt.show()
print("Autos con precio de agencia mayor a 30:", int((df["present_price"] > 30).sum()), "| precio de agencia máximo:", df["present_price"].max())""")
c.md("Hay un auto de 92.6 lakhs de agencia (un Land Cruiser) y 9 que pasan de 30. Son valores posibles y se conservan, pero ponen a prueba a cualquier modelo que no pueda extrapolar.")

c.md("""---
# Sección 3: Ajuste del Modelo
---

## Modelo de Machine Learning

Se separa el 20 % de los autos para la prueba.""")
c.code("""X, y = df[VARIABLES], df[OBJETIVO]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
print("Entrenamiento:", X_train.shape, " Prueba:", X_test.shape)""")
c.md("> Es importante siempre validar los rangos de los conjuntos creados, para evitar caer en extrapolación:")
c.code("""pd.DataFrame({"entrenamiento_max": X_train[NUMERICAS].max(), "prueba_max": X_test[NUMERICAS].max()})""")
c.md("""### Dos formas de predecir

1. **El precio** directamente.
2. **La razón** reventa / agencia, y luego el precio = razón por el precio de agencia.

Un árbol de decisión es constante por tramos: **nunca predice por encima del mayor valor que vio**. Si se entrena con el precio, no puede estimar un auto más caro que los del entrenamiento. La razón, en cambio, es una cantidad acotada, casi independiente de la escala del auto.

La clase `RazonAlPrecioActual` envuelve a cualquier modelo para que aprenda la razón (o su logaritmo).""")
c.code("""class RazonAlPrecioActual(BaseEstimator, RegressorMixin):
    def __init__(self, modelo, log=False):
        self.modelo = modelo
        self.log = log

    def fit(self, X, y):
        razon = np.asarray(y, dtype=float) / X["present_price"].to_numpy(dtype=float)
        self.modelo_ = clone(self.modelo).fit(X, np.log(razon) if self.log else razon)
        return self

    def predict(self, X):
        razon = self.modelo_.predict(X)
        return (np.exp(razon) if self.log else razon) * X["present_price"].to_numpy(dtype=float)""")
c.md("""### Candidatos

Hay dos **líneas base** (el precio medio y la regla "vale la razón mediana de su precio de agencia") y seis modelos. Cada uno es un `Pipeline` que incluye su preprocesamiento. La selección usa **solo el entrenamiento**, con validación cruzada repetida (5 particiones por 3 repeticiones) y el RMSE en lakhs.""")
c.code("""def armar(estimador):
    pre = ColumnTransformer([("num", StandardScaler(), NUMERICAS), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return Pipeline([("preprocesador", pre), ("modelo", estimador)])

bosque = lambda: RandomForestRegressor(n_estimators=200, random_state=SEMILLA, n_jobs=-1)
boosting = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=SEMILLA)

def candidatos():
    return {
        "precio medio (línea base)": armar(DummyRegressor(strategy="mean")),
        "regla de depreciación (línea base)": RazonAlPrecioActual(armar(DummyRegressor(strategy="median"))),
        "regresión lineal": armar(LinearRegression()),
        "random forest": armar(bosque()),
        "gradient boosting": armar(boosting()),
        "random forest (razón)": RazonAlPrecioActual(armar(bosque())),
        "gradient boosting (razón)": RazonAlPrecioActual(armar(boosting())),
        "regresión lineal (log-razón)": RazonAlPrecioActual(armar(LinearRegression()), log=True),
    }

validacion = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
filas = {}
for nombre, pipe in candidatos().items():
    r = cross_validate(pipe, X_train, y_train, cv=validacion, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
    filas[nombre] = {"RMSE cv": -r["test_rmse"].mean(), "desviación": r["test_rmse"].std(), "R2 cv": r["test_r2"].mean()}
comparacion = pd.DataFrame(filas).T.round(3)
comparacion""")
c.md("""Modelar la razón **reduce casi a la mitad el error** del bosque en niveles y, sobre todo, **estabiliza** el resultado: la desviación entre particiones baja mucho, porque en niveles el error depende de que un auto muy caro caiga en el pliegue de validación. Los dos modelos de razón con árboles no se distinguen entre sí.""")
c.code("""es_base = lambda nombre: "línea base" in nombre
ganador = comparacion[[not es_base(n) for n in comparacion.index]]["RMSE cv"].idxmin()
print("Modelo elegido:", ganador)
modelo = candidatos()[ganador].fit(X_train, y_train)""")

c.md("##### Evaluación en el conjunto de prueba")
c.code("""def metricas(real, pred):
    return {"R2": r2_score(real, pred), "MAE": mean_absolute_error(real, pred), "RMSE": np.sqrt(mean_squared_error(real, pred))}

prueba = {n: metricas(y_test, p.fit(X_train, y_train).predict(X_test)) for n, p in candidatos().items()}
pd.DataFrame(prueba).T.round(3)""")
c.md("""El bosque en niveles saca un R2 de prueba mucho menor que el de su validación cruzada. No es un fallo de la prueba sino de ese modelo, que no extrapola. Con **60 autos** de prueba, y **uno solo** de más de 15 lakhs, cualquier cifra es ruidosa. Por eso se calcula un intervalo de confianza.""")
c.code("""def intervalo_bootstrap(y, p, funcion, remuestreos=1000, semilla=42):
    rng = np.random.default_rng(semilla)
    valores = []
    while len(valores) < remuestreos:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            valores.append(funcion(y[i], p[i]))
    return [round(float(np.percentile(valores, 2.5)), 3), round(float(np.percentile(valores, 97.5)), 3)]

y_real, y_pred = np.asarray(y_test), modelo.predict(X_test)
print("IC 95 % del RMSE:", intervalo_bootstrap(y_real, y_pred, lambda a, b: float(np.sqrt(mean_squared_error(a, b)))))
print("IC 95 % del R2:  ", intervalo_bootstrap(y_real, y_pred, lambda a, b: float(r2_score(a, b))))

plt.scatter(y_test, y_pred, alpha=0.7)
limite = max(y_test.max(), y_pred.max())
plt.plot([0, limite], [0, limite], "r--")
plt.xlabel("Precio real (lakhs INR)")
plt.ylabel("Precio predicho (lakhs INR)")
plt.title("Real vs. predicho (prueba)")
plt.show()""")

c.md("""### Experimento: autos más caros que los del entrenamiento

Se entrena solo con los autos de precio de agencia de 15 lakhs o menos y se prueba con los más caros.""")
c.code("""barato = df["present_price"] <= 15
filas = {"autos de entrenamiento": int(barato.sum()), "autos caros de prueba": int((~barato).sum())}
resultados = {}
for nombre in ("random forest", ganador):
    ajustado = candidatos()[nombre].fit(df[barato][VARIABLES], df[barato][OBJETIVO])
    pred, real = ajustado.predict(df[~barato][VARIABLES]), df[~barato][OBJETIVO]
    resultados[nombre] = {"RMSE": np.sqrt(mean_squared_error(real, pred)), "sesgo (predicho - real)": (pred - real).mean()}
print(filas)
pd.DataFrame(resultados).T.round(2)""")
c.md("El bosque en niveles **subestima en promedio casi 6 lakhs** a los autos caros, porque no puede predecir más allá de lo que vio. El modelo de razón casi no tiene sesgo. Por eso la API puede estimar con sentido un auto de 90 lakhs de agencia.")

c.md("""### Modelo para servir y predicción

Con tan pocos datos, el modelo que se usa en la aplicación se reentrena con **todos** los autos.""")
c.code("""final = candidatos()[ganador].fit(X, y)
auto = pd.DataFrame([{"year": 2016, "present_price": 8.5, "kms_driven": 30000, "owner": 0,
                      "fuel_type": "Petrol", "seller_type": "Dealer", "transmission": "Manual"}])
valor = final.predict(auto[VARIABLES])[0]
print("Precio de reventa estimado: %.2f lakhs (unas %d rupias), %d %% del precio de agencia" % (valor, round(valor * 100_000), round(100 * valor / 8.5)))""")

c.md("""---
# Conclusiones
---

## Resultados

Con siete datos básicos del auto se puede estimar su precio de reventa con un error medio de unos 0.7 lakhs, frente a más de 3 al predecir el precio medio. El hallazgo principal es de modelado: **cambiar el objetivo de precio a razón** (reventa / agencia) casi duplica la precisión, estabiliza el resultado y permite estimar autos más caros que cualquiera del entrenamiento.

El hallazgo metodológico es que un R2 de prueba alto puede ser un accidente de la partición: con pocos datos hay que comparar todos los candidatos con la misma validación cruzada repetida y leer la prueba con su intervalo.

Limitaciones: son unos 300 autos de un solo mercado (India, 2003 a 2018). Los autos de gas natural y con tres dueños casi no tienen ejemplos. Con 60 autos de prueba las métricas son ruidosas, y las diferencias entre Random Forest y Gradient Boosting de razón no son concluyentes.

### Referencias

> Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
> Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.
> Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer.""")

if __name__ == "__main__":
    print(c.guardar("modelo_02_autos"))

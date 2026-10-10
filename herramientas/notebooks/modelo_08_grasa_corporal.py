"""Genera backend/features/modelo_08_grasa_corporal/notebook.ipynb (ver nb.py)."""
from nb import Cuaderno

c = Cuaderno()
c.portada(
    problema="""El porcentaje de grasa corporal es un indicador de salud más útil que el peso o el índice de masa corporal, pero su medición de referencia, el pesaje hidrostático, necesita equipo especializado. Queremos estimarlo con medidas que cualquier persona puede tomar con una balanza y una cinta métrica: edad, peso, estatura y diez circunferencias.

El asistente JarvisTEC usa este modelo para responder a la pregunta "qué porcentaje de grasa corporal tengo" a partir de los datos que el usuario escribe en un formulario. Es un problema de **regresión supervisada** y la variable objetivo es `BodyFat`, en porcentaje.""",
    titulo="Caso Aplicado: Porcentaje de Grasa Corporal",
    subtitulo="Regresión regularizada con medidas corporales",
    temas="Fuga de información · Multicolinealidad · Regresión lineal, Ridge y Lasso · Validación cruzada",
    herramientas="Python · Pandas · Scikit-learn · Seaborn",
    hilo=[("1 — Entendimiento y limpieza", "¿Qué forma tienen los datos y hay registros imposibles?"),
          ("2 — Exploración", "¿Qué medidas se relacionan con la grasa, y hay una variable que no se puede usar?"),
          ("3 — Modelo", "¿Qué regresión predice mejor y cómo se compara con predecir siempre el promedio?"),
          ("4 — Predicción", "¿Qué porcentaje de grasa se estima para una persona nueva?")],
    nota="el objetivo no es solo ajustar un buen modelo. También hay que reconocer qué datos no se pueden usar en la práctica y qué tan confiable es el resultado.",
    introduccion="Este notebook es un extra del modelo 08 del proyecto: reproduce de forma didáctica el entrenamiento que hace `train.py` y se puede probar en Google Colab. No reemplaza al código del repositorio.")

c.md("### Librerías")
c.code("""# Manejo de datasets
import pandas as pd
import numpy as np
# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Modelos
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Ignorar Warnings
import warnings
warnings.filterwarnings('ignore')

SEMILLA = 42""")

c.md("""---
# Sección 1: Entendimiento de los Datos
---

## Entendimiento de los Datos

El conjunto reúne 252 hombres adultos con medidas corporales. El porcentaje de grasa se estimó con pesaje hidrostático (Penrose et al., 1985; Johnson, 1996). Todas las variables son cuantitativas:

- BodyFat   : variable objetivo, porcentaje de grasa corporal
- Density   : densidad corporal (g/cm3), **no se usa**, se explica más adelante
- Age       : edad en años
- Weight    : peso en libras
- Height    : estatura en pulgadas
- Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist : circunferencias en centímetros""")
c.datos("modelo_08_grasa_corporal", "Son 252 filas y 15 columnas.")
c.code("""# Cargar los datos
datos = pd.read_csv(RUTA)
datos.head()""")
c.code("""# Describir el dataset
datos.describe().T""")
c.code("""# Información del dataset
datos.info()""")
c.code("""print("Cantidad de registros:", len(datos))
print("Cantidad de columnas:", len(datos.columns))
print("Tipo de datos:", datos.dtypes.unique())
print("Filas y columnas:", datos.shape)
print("Valores nulos:", int(datos.isnull().sum().sum()))""")

c.md("""El formulario de la aplicación usa las unidades que la gente conoce, así que se pasa el peso a kilogramos y la estatura a centímetros, y se renombran las columnas.""")
c.code("""LB_A_KG = 0.45359237
PULGADA_A_CM = 2.54

df = datos.copy()
df.columns = [c.lower() for c in df.columns]
df["weight_kg"] = (df.pop("weight") * LB_A_KG).round(2)
df["height_cm"] = (df.pop("height") * PULGADA_A_CM).round(1)
df = df.rename(columns={c: f"{c}_cm" for c in
                        ["neck", "chest", "abdomen", "hip", "thigh", "knee", "ankle", "biceps", "forearm", "wrist"]})
df.head()""")

c.md("""### Registros imposibles

Es importante revisar los extremos antes de entrenar. Se descartan solo los registros físicamente imposibles: una persona con 0 % de grasa y otra con 75 cm de estatura (un error de digitación evidente). Los dos casos coinciden con errores que ya señala la literatura del dataset (Johnson, 1996). Los demás extremos son posibles y se conservan.""")
c.code("""imposibles = (df["bodyfat"] <= 0) | (df["height_cm"] < 120)
print("Registros imposibles descartados:", int(imposibles.sum()), "-> filas", df.index[imposibles].tolist())
df = df[~imposibles].reset_index(drop=True)
print("Quedan", len(df), "registros")""")

c.md("""---
# Sección 2: Exploración y Selección de Predictores
---

## Exploración de los Datos""")
c.code("""sns.histplot(df["bodyfat"], kde=True)
plt.xlabel("Grasa corporal (%)")
plt.title("Distribución del porcentaje de grasa corporal")
plt.show()
print("Media: %.1f %%  Desviación: %.1f" % (df["bodyfat"].mean(), df["bodyfat"].std()))""")
c.md("La distribución es aproximadamente simétrica, con una media cercana al 19 %.")

VARS = '["age", "weight_kg", "height_cm", "neck_cm", "chest_cm", "abdomen_cm", "hip_cm", "thigh_cm", "knee_cm", "ankle_cm", "biceps_cm", "forearm_cm", "wrist_cm"]'
c.code(f"""VARIABLES = {VARS}
OBJETIVO = "bodyfat"

corr = df[VARIABLES + [OBJETIVO]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={{"size": 7}})
plt.title("Correlación entre variables")
plt.show()
corr[OBJETIVO].drop(OBJETIVO).sort_values().round(2)""")
c.md("""> Analogía: cuando dos variables predictoras están muy correlacionadas entre sí, es como pedirle a dos testigos que vieron exactamente lo mismo que declaren por separado. El modelo no puede distinguir el aporte real de cada uno y sus coeficientes se vuelven inestables.

**Multicolinealidad**

La circunferencia del abdomen es la variable más asociada a la grasa (r = 0.81), seguida del pecho, la cadera y el peso. La estatura casi no se relaciona. Además, las circunferencias están muy correlacionadas entre sí (peso y cadera, r = 0.94). Esto es un argumento para probar modelos con regularización (Ridge y Lasso).""")
c.code("""sns.regplot(data=df, x="abdomen_cm", y=OBJETIVO, scatter_kws={"alpha": 0.6})
plt.title("Circunferencia abdominal vs. grasa corporal")
plt.show()""")
c.md("La relación entre el abdomen y la grasa es aproximadamente lineal, lo que justifica empezar con modelos lineales.")

c.md("""### Una variable que no se puede usar: Density

`BodyFat` no es una medición independiente: se calcula a partir de la densidad corporal con la ecuación de Siri (1956). La correlación entre las dos es de -0.99. Veamos qué pasa si se incluye.""")
c.code("""print("Correlación Density vs BodyFat: %.3f" % df["density"].corr(df["bodyfat"]))

X_tr, X_te, y_tr, y_te = train_test_split(df[VARIABLES + ["density"]], df[OBJETIVO], test_size=0.2, random_state=SEMILLA)
con_densidad = Pipeline([("escala", StandardScaler()), ("modelo", LinearRegression())]).fit(X_tr, y_tr)
print("R2 de prueba con Density: %.3f" % r2_score(y_te, con_densidad.predict(X_te)))""")
c.md("""El R2 de 0.99 es engañoso: la variable ya contiene la respuesta. Medir la densidad exige el mismo pesaje hidrostático que se quiere evitar, así que **se descarta**. A esto se le llama fuga de información.""")

c.md("""---
# Sección 3: Ajuste del Modelo
---

## Modelo de Machine Learning

Primero se divide el conjunto en 80 % para entrenamiento y 20 % para prueba.""")
c.code("""X, y = df[VARIABLES], df[OBJETIVO]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
print("Entrenamiento:", X_train.shape, " Prueba:", X_test.shape)""")
c.md("> Es importante siempre validar los rangos de los conjuntos creados, para evitar caer en extrapolación:")
c.code("""pd.DataFrame({"entrenamiento_min": X_train.min(), "entrenamiento_max": X_train.max(),
              "prueba_min": X_test.min(), "prueba_max": X_test.max()}).round(1)""")
c.md("""Hay una persona en la prueba con 165 kg y 148 cm de abdomen, mucho más que cualquiera del entrenamiento (119 kg y 126 cm como máximo). Los modelos lineales extrapolan mal fuera de lo que vieron, y eso se nota en la evaluación.

### Candidatos

Cada candidato es un `Pipeline` que incluye su preprocesamiento. La selección se hace **solo con el entrenamiento**, con validación cruzada repetida (5 particiones por 3 repeticiones). La prueba se usa una sola vez.""")
c.code("""def con_escala(estimador):
    return Pipeline([("escala", StandardScaler()), ("modelo", estimador)])

candidatos = {
    "media (línea base)": Pipeline([("modelo", DummyRegressor(strategy="mean"))]),
    "regresión lineal": con_escala(LinearRegression()),
    "ridge": con_escala(RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "lasso": con_escala(LassoCV(cv=5, random_state=SEMILLA, max_iter=20000)),
    "random forest": Pipeline([("modelo", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEMILLA, n_jobs=-1))]),
}

validacion = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
filas = {}
for nombre, pipe in candidatos.items():
    rmse = -cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="neg_root_mean_squared_error")
    r2 = cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="r2")
    filas[nombre] = {"RMSE cv": rmse.mean(), "desviación": rmse.std(), "R2 cv": r2.mean()}
comparacion = pd.DataFrame(filas).T.round(3)
comparacion""")
c.md("""Las diferencias entre regresión lineal, Ridge y Lasso son mucho menores que la variación entre particiones (alrededor de 0.4), así que no se pueden distinguir. Se elige Lasso por tener el menor RMSE y por dar un modelo más simple, no porque supere de forma demostrable a los otros.""")
c.code("""ganador = comparacion.drop("media (línea base)")["RMSE cv"].idxmin()
print("Modelo elegido:", ganador)
modelo = candidatos[ganador].fit(X_train, y_train)""")

c.md("##### Evaluación en el conjunto de prueba")
c.code("""y_pred = modelo.predict(X_test)
base = candidatos["media (línea base)"].fit(X_train, y_train).predict(X_test)

def metricas(real, pred):
    return {"R2": r2_score(real, pred), "MAE": mean_absolute_error(real, pred), "RMSE": np.sqrt(mean_squared_error(real, pred))}

pd.DataFrame({"media (línea base)": metricas(y_test, base), ganador: metricas(y_test, y_pred)}).T.round(3)""")
c.code("""plt.scatter(y_test, y_pred, alpha=0.7)
limite = max(y_test.max(), y_pred.max())
plt.plot([0, limite], [0, limite], "r--")
plt.xlabel("Grasa corporal real (%)")
plt.ylabel("Grasa corporal predicha (%)")
plt.title("Real vs. predicho (prueba)")
plt.show()""")
c.md("""El R2 de prueba es menor que el de validación cruzada. La causa principal es la persona de 165 kg: el modelo estimó cerca de 55 % cuando el valor real era 35 %. Con solo 50 filas de prueba, un único registro mueve mucho la métrica.""")
c.code("""lasso = modelo.named_steps["modelo"]
coeficientes = pd.Series(lasso.coef_, index=VARIABLES).round(2)
print("alpha elegido: %.3f" % lasso.alpha_)
coeficientes[coeficientes != 0].sort_values()""")
c.md("Lasso conserva pocas variables y anula las demás. Con las variables estandarizadas, el abdomen domina por mucho.")

c.md("### Predicción para una persona nueva")
c.code("""persona = pd.DataFrame([{"age": 35, "weight_kg": 80.0, "height_cm": 178.0, "neck_cm": 38.0, "chest_cm": 100.0,
                         "abdomen_cm": 90.0, "hip_cm": 99.0, "thigh_cm": 58.0, "knee_cm": 38.0, "ankle_cm": 23.0,
                         "biceps_cm": 32.0, "forearm_cm": 28.0, "wrist_cm": 17.5}])
print("Grasa corporal estimada: %.1f %%" % modelo.predict(persona[VARIABLES])[0])""")

c.md("""---
# Conclusiones
---

## Resultados

Con medidas corporales simples es posible estimar la grasa corporal con un error medio de aproximadamente 4 puntos porcentuales, frente a 6.5 al predecir siempre el promedio. La circunferencia abdominal concentra casi toda la información útil: Lasso rinde igual con unas pocas variables que con las trece.

El hallazgo metodológico más importante es la fuga de información de `Density`. Un R2 de 0.99 parece excelente, pero no se puede usar en la práctica.

Limitaciones: el conjunto tiene solo 252 hombres adultos, así que el modelo no es válido para mujeres ni para menores. Las estimaciones son orientativas y no diagnósticas. Con medidas extremas el modelo extrapola mal, y por eso la API del proyecto avisa cuando una entrada sale del rango de entrenamiento.

### Referencias

> Johnson, R. W. (1996). Fitting percentage of body fat to simple body measurements. *Journal of Statistics Education*, 4(1).
> Penrose, K. W., Nelson, A. G., & Fisher, A. G. (1985). Generalized body composition prediction equation for men using simple measurement techniques. *Medicine & Science in Sports & Exercise*, 17(2), 189.
> Siri, W. E. (1956). The gross composition of the body. En *Advances in Biological and Medical Physics* (Vol. 4). Academic Press.
> Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.""")

if __name__ == "__main__":
    print(c.guardar("modelo_08_grasa_corporal"))

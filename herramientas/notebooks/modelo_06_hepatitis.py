"""Genera backend/features/modelo_06_hepatitis/notebook.ipynb (ver nb.py)."""
from nb import Cuaderno

c = Cuaderno()
c.portada(
    problema="""La infección crónica por el virus de la hepatitis C puede evolucionar de una hepatitis a fibrosis y a cirrosis, y la etapa se estima con análisis de sangre y otras pruebas (Manns et al., 2017). Queremos clasificar el estado hepático de una persona en cuatro clases ordenadas: **donante** (sano), **hepatitis**, **fibrosis** y **cirrosis**, a partir de su edad y de diez análisis de laboratorio.

Es un problema de **clasificación multiclase supervisada muy desbalanceada**: cerca del 88 % son donantes y cada clase de enfermedad tiene entre 21 y 30 casos.

**Es un ejercicio educativo con un conjunto de datos pequeño. No es una herramienta de diagnóstico ni reemplaza a un profesional de la salud.**""",
    titulo="Caso Aplicado: Estado Hepático por Hepatitis C",
    subtitulo="Clasificación multiclase con clases muy raras, datos faltantes y poblaciones distintas",
    temas="Clases raras · Valores faltantes: artefacto o señal · Validación cruzada estratificada · Intervalos bootstrap",
    herramientas="Python · Pandas · Scikit-learn · Seaborn",
    hilo=[("1 — Entendimiento y limpieza", "¿Qué clases hay, cuáles son ambiguas y qué dicen los valores faltantes?"),
          ("2 — Exploración", "¿Qué análisis cambian con la gravedad de la enfermedad?"),
          ("3 — Modelo", "¿Qué clasificador rinde mejor con tan pocos casos de cada enfermedad?"),
          ("4 — Evaluación", "¿Qué tan estable es el resultado y qué clases se detectan mal?")],
    nota="con 4 a 6 casos por clase en la prueba, el resultado depende mucho de qué personas cayeron ahí. Hay que mirar las predicciones fuera de muestra y los intervalos, no solo un número.",
    introduccion="Este notebook es un extra del modelo 06 del proyecto: reproduce de forma didáctica el entrenamiento que hace `train.py` y se puede probar en Google Colab. No reemplaza al código del repositorio.")

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

El conjunto reúne datos de laboratorio de donantes de sangre y de pacientes con hepatitis C (Lichtinghagen et al., 2013; Hoffmann et al., 2018). Tiene 615 personas y estas columnas:

- Category : variable objetivo, la clase (donante, "suspect Blood Donor", hepatitis, fibrosis o cirrosis)
- Age, Sex : edad y sexo
- ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT : diez análisis de laboratorio, todas variables cuantitativas
- La primera columna sin nombre es un índice y se descarta""")
c.datos("modelo_06_hepatitis", "Son 615 filas y 14 columnas.")
c.code("""# Cargar los datos
datos = pd.read_csv(RUTA)
datos.head()""")
c.code("""# Información del dataset
datos.info()""")
c.md("""### Limpieza

Siete personas tienen la categoría "suspect Blood Donor". No se parecen a un donante sano ni corresponden a una etapa de la enfermedad, así que se **excluyen**.""")
c.code("""print(datos["Category"].value_counts().to_dict())
print("Albúmina mediana de los 'suspect':", datos[datos["Category"] == "0s=suspect Blood Donor"]["ALB"].median(),
      "| de los donantes:", datos[datos["Category"] == "0=Blood Donor"]["ALB"].median())

CLASES = {"0=Blood Donor": "donante", "1=Hepatitis": "hepatitis", "2=Fibrosis": "fibrosis", "3=Cirrhosis": "cirrosis"}
ORDEN = ["donante", "hepatitis", "fibrosis", "cirrosis"]
ENFERMEDAD = ["hepatitis", "fibrosis", "cirrosis"]

df = datos.drop(columns=datos.columns[0])
df.columns = [c.lower() for c in df.columns]
df = df[df["category"].isin(CLASES)].reset_index(drop=True)
df["clase"] = df.pop("category").map(CLASES)
print("\\nQuedan", len(df), "personas:", df["clase"].value_counts().reindex(ORDEN).to_dict())""")

c.md("""### Valores faltantes: ¿artefacto o señal?

Hay 18 valores vacíos de ALP. Se revisa en qué clases están.""")
c.code("""LABORATORIO = ["alb", "alp", "alt", "ast", "bil", "che", "chol", "crea", "ggt", "prot"]
print("Nulos:", df.isna().sum()[lambda s: s > 0].to_dict())
print("ALP vacío por clase:", df[df["alp"].isna()]["clase"].value_counts().reindex(ORDEN, fill_value=0).to_dict())
print("CHOL vacío por clase:", df[df["chol"].isna()]["clase"].value_counts().reindex(ORDEN, fill_value=0).to_dict())
df["alp_faltante"] = df["alp"].isna().astype(int)""")
c.md("""Todos los ALP vacíos están en pacientes y ninguno en donantes. Eso hace sospechar un artefacto de la recolección, como el del `bmi` en el modelo de ACV. **Pero no hay que concluir por analogía: se comprueba** (más adelante, en la sección del modelo).

También hay una diferencia de población: veamos la edad.""")
c.code("""print("Edad mínima por clase:", df.groupby("clase")["age"].min().reindex(ORDEN).to_dict())
print("Edad mediana por clase:", df.groupby("clase")["age"].median().reindex(ORDEN).to_dict())""")
c.md("Ningún donante tiene menos de 32 años, mientras que los pacientes llegan hasta los 19. Donantes y pacientes vienen de poblaciones distintas, y el modelo podría aprender diferencias de población y no solo de enfermedad.")

c.md("""---
# Sección 2: Exploración de los Datos
---

## Exploración de los Datos""")
c.code("""sns.countplot(data=df, x="clase", order=ORDEN)
plt.title("Distribución de las clases")
plt.show()""")
c.code("""fig, ejes = plt.subplots(2, 4, figsize=(15, 7))
for eje, variable in zip(ejes.ravel(), ["alb", "ast", "alt", "ggt", "bil", "che", "chol", "age"]):
    sns.boxplot(data=df, x="clase", y=variable, order=ORDEN, ax=eje)
    eje.set_title(variable)
    eje.set_xlabel("")
    eje.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()
df.groupby("clase")[["ast", "ggt", "alb", "che", "bil"]].median().reindex(ORDEN).round(1)""")
c.md("""Las enzimas AST y GGT **suben con la gravedad**. En la cirrosis bajan la albúmina y la colinesterasa y sube la bilirrubina. Todo esto es coherente con lo que se espera de la fisiología del hígado (Giannini et al., 2005). La hepatitis es la clase más difícil de separar de un donante, porque su laboratorio es casi normal salvo por las enzimas.""")
c.code("""plt.figure(figsize=(8, 6.5))
sns.heatmap(df[["age"] + LABORATORIO].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlación entre variables")
plt.show()""")
c.md("AST, ALT y GGT están correlacionadas entre sí, es decir, aportan información parcialmente repetida.")

c.md("""---
# Sección 3: Ajuste del Modelo
---

## Modelo de Machine Learning

La división es estratificada. Con solo 21 a 30 casos por clase de enfermedad, cada uno cuenta.""")
c.code("""NUMERICAS = ["age"] + LABORATORIO
VARIABLES = NUMERICAS   # edad y diez análisis; el sexo no aporta nada medible y es un dato sensible
train, test, y_train, y_test = train_test_split(df, df["clase"], test_size=0.2, stratify=df["clase"], random_state=SEMILLA)
print("Entrenamiento:", len(train), " Prueba:", len(test))
print("Casos por clase en la prueba:", y_test.value_counts().reindex(ORDEN).to_dict())""")
c.md("""### El ALP: artefacto o señal

Si el aporte del ALP viniera de sus vacíos, entonces (1) rellenar los vacíos con **valores observados al azar** lo haría desaparecer y (2) un **indicador de "ALP faltante"** lo reproduciría. Se prueba con un Random Forest y validación cruzada (solo con el entrenamiento).""")
c.code("""def armar(estimador, variables=VARIABLES, escalar=True):
    categoricas = [v for v in variables if v == "sex"]
    numericas = [v for v in variables if v != "sex"]
    columnas = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else [])), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])

validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMILLA)
bosque = dict(n_estimators=200, min_samples_leaf=2, max_depth=8, class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=-1)

azar = train.copy()
vacios = azar["alp"].isna()
azar.loc[vacios, "alp"] = np.random.default_rng(SEMILLA).choice(train["alp"].dropna().to_numpy(), int(vacios.sum()))
sin_alp = [v for v in VARIABLES if v != "alp"]
casos = {
    "con ALP": (train, VARIABLES),
    "sin ALP": (train, sin_alp),
    "con los vacíos de ALP rellenados al azar": (azar, VARIABLES),
    "solo el indicador de ALP faltante (sin ALP)": (train, sin_alp + ["alp_faltante"]),
    "solo filas con ALP observada, con ALP": (train[train["alp"].notna()], VARIABLES),
    "solo filas con ALP observada, sin ALP": (train[train["alp"].notna()], sin_alp),
}
filas = {}
for nombre, (datos_cv, cols) in casos.items():
    r = cross_validate(armar(RandomForestClassifier(**bosque), cols, escalar=False), datos_cv[cols], datos_cv["clase"], cv=validacion, scoring="f1_macro")
    filas[nombre] = {"F1 macro cv": r["test_score"].mean(), "desviación": r["test_score"].std()}
pd.DataFrame(filas).T.round(3)""")
c.md("""La mejora se mantiene con el relleno aleatorio, el indicador solo aporta poco, y en las filas donde el ALP sí se observó la mejora es mayor. Es lo contrario de lo que pasaba con el `bmi` del modelo de ACV: **aquí el ALP aporta señal real**, no un artefacto. Se incluye, imputado con la mediana dentro del Pipeline.

> Analogía: dos pacientes con el mismo síntoma no se tratan igual solo porque un caso parezca "como el anterior". Primero se hace el examen. Aquí el examen mostró que el ALP sí informa, y que el `bmi` no.

### Candidatos

Se ponderan las clases para que las raras cuenten. Por eso los puntajes del modelo **no son probabilidades calibradas**.""")
c.code("""def candidatos():
    return {
        "clase mayoritaria (línea base)": armar(DummyClassifier(strategy="most_frequent")),
        "regresión logística": armar(LogisticRegression(max_iter=5000, class_weight="balanced")),
        "k vecinos": armar(KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "random forest": armar(RandomForestClassifier(**bosque), escalar=False),
    }

X_train, X_test = train[VARIABLES], test[VARIABLES]
filas = {}
for nombre, pipe in candidatos().items():
    r = cross_validate(pipe, X_train, y_train, cv=validacion, scoring={"f1": "f1_macro", "bal": "balanced_accuracy"})
    filas[nombre] = {"F1 macro cv": r["test_f1"].mean(), "desviación": r["test_f1"].std(), "exactitud balanceada cv": r["test_bal"].mean()}
comparacion = pd.DataFrame(filas).T.round(3)
comparacion""")
c.md("""La desviación entre particiones es grande (alrededor de 0.07 a 0.11) y las diferencias entre la regresión logística y el Random Forest son menores que eso. Se elige el de mayor F1 macro, pero la elección es casi un empate.""")
c.code("""ganador = comparacion.drop("clase mayoritaria (línea base)")["F1 macro cv"].idxmax()
print("Modelo elegido:", ganador)
modelo = candidatos()[ganador].fit(X_train, y_train)""")

c.md("##### Evaluación en el conjunto de prueba")
c.code("""pred = modelo.predict(X_test)
print(classification_report(y_test, pred, labels=ORDEN, digits=3, zero_division=0))
matriz = confusion_matrix(y_test, pred, labels=ORDEN)
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión (prueba)")
plt.show()""")
c.code("""def sensibilidad_y_especificidad(real, predicho):
    r, p = np.isin(np.asarray(real), ENFERMEDAD), np.isin(np.asarray(predicho), ENFERMEDAD)
    return {"sensibilidad": round(float(p[r].mean()), 3), "especificidad": round(float((~p[~r]).mean()), 3)}

print("Enfermedad contra donante (prueba):", sensibilidad_y_especificidad(y_test, pred))
f1_macro_de = lambda y, p: float(f1_score(y, p, labels=ORDEN, average="macro", zero_division=0))

def intervalo_bootstrap(y, p, funcion, remuestreos=1000, semilla=42):
    rng = np.random.default_rng(semilla)
    valores = []
    while len(valores) < remuestreos:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            valores.append(funcion(y[i], p[i]))
    return [round(float(np.percentile(valores, 2.5)), 4), round(float(np.percentile(valores, 97.5)), 4)]

print("F1 macro de prueba: %.3f" % f1_macro_de(y_test, pred))
print("IC 95 % del F1 macro:", intervalo_bootstrap(np.asarray(y_test), pred, f1_macro_de))""")
c.md("""La exactitud es alta, pero engaña: el 88 % son donantes. El intervalo del F1 macro es muy amplio, porque en la prueba hay solo 4 a 6 casos de cada enfermedad. Una sola persona cambia el resultado de forma importante.

### Predicciones fuera de muestra del entrenamiento

Para tener estimaciones por clase menos ruidosas se usan las predicciones de validación cruzada sobre las 486 personas del entrenamiento.""")
c.code("""oof = cross_val_predict(candidatos()[ganador], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
informe = pd.DataFrame(classification_report(y_train, oof, labels=ORDEN, output_dict=True, zero_division=0)).T.loc[ORDEN].round(3)
print("F1 macro fuera de muestra: %.3f" % f1_macro_de(y_train, oof))
print("Enfermedad contra donante (fuera de muestra):", sensibilidad_y_especificidad(y_train, oof))
informe""")
c.md("""La cirrosis se reconoce bien. **La hepatitis y la fibrosis se detectan mal**, algo esperable porque sus análisis de laboratorio se parecen a los de un donante.

La regresión logística, casi empatada en validación cruzada, tiene otro compromiso entre sensibilidad y especificidad. Vale la pena mirarlo:""")
c.code("""oof_lr = cross_val_predict(candidatos()["regresión logística"], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
print("Regresión logística, F1 macro fuera de muestra: %.3f" % f1_macro_de(y_train, oof_lr))
print("Enfermedad contra donante:", sensibilidad_y_especificidad(y_train, oof_lr))""")
c.md("La regresión logística detecta mucho más la enfermedad (mayor sensibilidad), a costa de confundir más donantes con enfermos (menor especificidad). Elegir entre las dos es decidir qué error importa más.")

c.md("### Predicción para una persona nueva")
c.code("""persona = pd.DataFrame([{"age": 47, "alb": 42.0, "alp": 66.0, "alt": 23.0, "ast": 25.0, "bil": 7.3, "che": 8.3, "chol": 5.3, "crea": 77.0, "ggt": 23.0, "prot": 72.0}])
puntajes = dict(zip(modelo.classes_, modelo.predict_proba(persona[VARIABLES])[0].round(3)))
print("Clase más probable:", modelo.predict(persona[VARIABLES])[0])
print({k: float(v) for k, v in puntajes.items()})""")

c.md("""---
# Conclusiones
---

## Resultados

Con la edad y diez análisis de sangre se distingue bien a los donantes de los pacientes y se reconoce la cirrosis, pero **la hepatitis y la fibrosis se detectan mal**. Las enzimas AST y GGT, la bilirrubina, la albúmina, la colinesterasa y el ALP son las señales más claras.

Hay tres hallazgos metodológicos. Primero, ante un valor faltante sospechoso hay que distinguir artefacto de señal con un experimento: aquí el ALP resultó ser señal. Segundo, la elección entre dos modelos casi empatados es en realidad un compromiso entre sensibilidad y especificidad que debe declararse. Tercero, con 4 a 6 casos por clase en la prueba el resultado depende mucho de la partición.

Limitaciones: **no es un diagnóstico**. Hay solo 21 a 30 casos por clase de enfermedad. Donantes y pacientes vienen de poblaciones distintas (ningún donante tiene menos de 32 años), así que el modelo puede aprender la población y no la enfermedad, y no hay validación externa. Los puntajes no son probabilidades calibradas. Las unidades de los análisis no vienen documentadas en la fuente.

### Referencias

> Giannini, E. G., Testa, R., & Savarino, V. (2005). Liver enzyme alteration: a guide for clinicians. *Canadian Medical Association Journal*, 172(3), 367-379.
> Hoffmann, G., Bietenbeck, A., Lichtinghagen, R., & Klawonn, F. (2018). Using machine learning techniques to generate laboratory diagnostic pathways: a case study. *Journal of Laboratory and Precision Medicine*, 3.
> Lichtinghagen, R., Pietsch, D., Bantel, H., Manns, M. P., Brand, K., & Bahr, M. J. (2013). The Enhanced Liver Fibrosis (ELF) score: Normal values, influence factors and proposed cut-off values. *Journal of Hepatology*, 59(2), 236-242.
> Manns, M. P., Buti, M., Gane, E., et al. (2017). Hepatitis C virus infection. *Nature Reviews Disease Primers*, 3, 17006.""")

if __name__ == "__main__":
    print(c.guardar("modelo_06_hepatitis"))

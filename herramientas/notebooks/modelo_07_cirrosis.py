"""Genera backend/features/modelo_07_cirrosis/notebook.ipynb (ver nb.py)."""
from nb import Cuaderno

c = Cuaderno()
c.portada(
    problema="""La cirrosis biliar primaria es una enfermedad hepática crónica cuya gravedad se clasifica en cuatro **etapas histológicas**, de la 1 (lesión temprana) a la 4 (cirrosis), mediante biopsia (Ludwig et al., 1978). La biopsia es invasiva, así que queremos **estimar la etapa a partir de signos clínicos y análisis de laboratorio** de la consulta inicial.

Es un problema de **clasificación multiclase ordinal supervisada**: equivocarse por tres etapas es peor que equivocarse por una. Por eso, además de las métricas habituales, se usan el error medio en etapas, la proporción de aciertos con error de a lo sumo una etapa y el kappa cuadrático ponderado (Cohen, 1968).

**Es un ejercicio educativo con pocos pacientes. No es una herramienta de diagnóstico: la etapa se determina por biopsia.**""",
    titulo="Caso Aplicado: Etapa de la Cirrosis Biliar Primaria",
    subtitulo="Clasificación ordinal con pocos pacientes, fuga de información y datos faltantes",
    temas="Clasificación ordinal · Fuga de información · Pacientes incompletos · Qué variables usa realmente el modelo",
    herramientas="Python · Pandas · Scikit-learn · Seaborn",
    hilo=[("1 — Entendimiento y limpieza", "¿Qué variables existen en la consulta inicial y cuáles son seguimiento posterior?"),
          ("2 — Exploración", "¿Qué signos y análisis cambian con la etapa?"),
          ("3 — Modelo", "¿Qué clasificador rinde mejor y cómo se compara con predecir siempre la etapa más común?"),
          ("4 — Evaluación", "¿Qué tan lejos se equivoca el modelo y qué variables usa de verdad?")],
    nota="describir los datos no es medir lo que el modelo usa. Una variable puede verse muy distinta entre etapas y aun así no aportar nada al modelo, por eso se mide quitando cada variable.",
    introduccion="Este notebook es un extra del modelo 07 del proyecto: reproduce de forma didáctica el entrenamiento que hace `train.py` y se puede probar en Google Colab. No reemplaza al código del repositorio.")

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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, cross_validate, train_test_split
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

El conjunto viene del ensayo clínico de la Clínica Mayo sobre cirrosis biliar primaria, 1974 a 1984 (Dickson et al., 1989). Tiene 418 pacientes y 20 columnas:

- ID, N_Days, Status, Drug : identificador, días de seguimiento, estado final y tratamiento
- Age (en días), Sex      : edad y sexo
- Ascites, Hepatomegaly, Spiders, Edema : signos clínicos
- Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin : análisis de laboratorio, todas variables cuantitativas
- Stage : variable objetivo, la etapa histológica de 1 a 4""")
c.datos("modelo_07_cirrosis", "Son 418 filas y 20 columnas.")
c.code("""# Cargar los datos
datos = pd.read_csv(RUTA)
datos.head()""")
c.code("""# Información del dataset
datos.info()""")
c.md("""### Limpieza

La edad viene en días y se pasa a años. Seis pacientes no tienen etapa y se descartan. Las columnas `ID` y `Drug` también se descartan: el tratamiento fue aleatorizado, no depende de la etapa y está vacío en los pacientes fuera del ensayo.""")
c.code("""df = datos.drop(columns=["ID", "Drug"])
df.columns = [c.lower() for c in df.columns]
df["age"] = df["age"] / 365.25
df = df.dropna(subset=["stage"]).reset_index(drop=True)
df["etapa"] = df.pop("stage").astype(int)

print("Pacientes con etapa:", len(df))
print("Etapas:", df["etapa"].value_counts().sort_index().to_dict())
print("Edad (años): mínimo %.1f, máximo %.1f, media %.1f" % (df["age"].min(), df["age"].max(), df["age"].mean()))
print("Nulos:", df.isna().sum()[lambda s: s > 0].to_dict())""")
c.md("""### Pacientes incompletos

312 pacientes participaron en el ensayo y tienen datos casi completos. Los otros 106 no participaron y solo tienen las mediciones básicas. Antes de decidir qué hacer con ellos, se comprueba si los vacíos dependen de la etapa.""")
c.code("""NUMERICAS = ["age", "bilirubin", "cholesterol", "albumin", "copper", "alk_phos", "sgot", "tryglicerides", "platelets", "prothrombin"]
CATEGORICAS = ["sex", "ascites", "hepatomegaly", "spiders", "edema"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "etapa"

sin_ensayo = df[df["ascites"].isna()]
print("Pacientes fuera del ensayo (sin ascitis, cobre, SGOT):", len(sin_ensayo))
print("Etapas fuera del ensayo:", sin_ensayo[OBJETIVO].value_counts(normalize=True).sort_index().round(2).to_dict())
print("Etapas en el resto:     ", df[df["ascites"].notna()][OBJETIVO].value_counts(normalize=True).sort_index().round(2).to_dict())

completos = df.dropna(subset=VARIABLES).reset_index(drop=True)
print("\\nPacientes con las 15 variables:", len(completos), "| etapas:", completos[OBJETIVO].value_counts().sort_index().to_dict())""")
c.md("""Los vacíos **no dependen de la etapa**: la distribución es casi igual dentro y fuera del ensayo. Por eso aquí no hay un artefacto como el del `bmi` en el modelo de ACV. Aun así, en el trabajo del repositorio se comprobó que los pacientes incompletos no ayudan al modelo, y se entrena solo con los **276 pacientes completos**.

### Una variable que no se puede usar: el seguimiento posterior

`N_Days` y `Status` son el seguimiento posterior (cuántos días vivió el paciente y si falleció). No existen en la consulta inicial: un asistente no puede preguntar por lo que ocurrirá después. Se mide si inflarían la métrica.""")
c.code("""print("Mediana de N_Days por etapa:", df.groupby(OBJETIVO)["n_days"].median().to_dict())
print("Fallecidos por etapa:", df.groupby(OBJETIVO)["status"].apply(lambda s: round((s == "D").mean(), 2)).to_dict())""")
c.md("Los pacientes de etapas avanzadas tienen seguimientos más cortos y fallecen más, lo cual es esperable. Veamos qué tanta información aportan al modelo, más adelante.")

c.md("""---
# Sección 2: Exploración de los Datos
---

## Exploración de los Datos

Las figuras usan los 276 pacientes completos.""")
c.code("""sns.countplot(data=completos, x=OBJETIVO)
plt.title("Pacientes completos por etapa")
plt.show()""")
c.md("La etapa 1 es muy rara: hay apenas 12 pacientes completos. Eso va a hacer ruidosa cualquier métrica de esa clase.")
c.code("""fig, ejes = plt.subplots(2, 4, figsize=(15, 7))
for eje, variable in zip(ejes.ravel(), ["bilirubin", "albumin", "copper", "platelets", "prothrombin", "alk_phos", "sgot", "age"]):
    sns.boxplot(data=completos, x=OBJETIVO, y=variable, ax=eje)
    eje.set_title(variable)
    if variable in ("copper", "alk_phos", "bilirubin"):
        eje.set_yscale("log")
plt.tight_layout()
plt.show()
completos.groupby(OBJETIVO)[["bilirubin", "albumin", "platelets", "prothrombin"]].median().round(2)""")
c.code("""fig, ejes = plt.subplots(1, 4, figsize=(14, 3.6))
for eje, variable in zip(ejes, ["hepatomegaly", "spiders", "ascites", "edema"]):
    pd.crosstab(completos[OBJETIVO], completos[variable], normalize="index").plot.bar(stacked=True, ax=eje, legend=variable == "edema")
    eje.set_title(variable)
    eje.set_xlabel("etapa")
plt.tight_layout()
plt.show()""")
c.md("""Los indicios de gravedad **crecen con la etapa**: sube la bilirrubina, baja la albúmina y aparecen más la hepatomegalia, los angiomas en araña, la ascitis y el edema. Las etapas intermedias (2 y 3) son difíciles de separar porque sus medianas se parecen.

> Analogía: es como separar las notas de un examen en cuatro grupos. Los extremos se distinguen fácil, pero entre el grupo 2 y el 3 casi nadie sabe dónde está la frontera.""")

c.md("""---
# Sección 3: Ajuste del Modelo
---

## Modelo de Machine Learning

La división es estratificada sobre los pacientes completos.""")
c.code("""train, test, y_train, y_test = train_test_split(completos, completos[OBJETIVO], test_size=0.2, stratify=completos[OBJETIVO], random_state=SEMILLA)
X_train, X_test = train[VARIABLES], test[VARIABLES]
print("Entrenamiento:", len(train), "| etapas:", y_train.value_counts().sort_index().to_dict())
print("Prueba:", len(test), "| etapas:", y_test.value_counts().sort_index().to_dict())""")
c.md("""Con solo 2 pacientes de etapa 1 en la prueba, esa clase no permite sacar conclusiones.

### Candidatos

Las clases están ponderadas (la etapa 1 tiene solo 9 pacientes en el entrenamiento), así que los puntajes **no son probabilidades calibradas**. La selección usa solo el entrenamiento, con validación cruzada estratificada repetida (5 por 4) y F1 macro.""")
c.code("""def armar(estimador, variables=VARIABLES, escalar=True):
    categoricas = [v for v in variables if v in CATEGORICAS + ["status"]]
    numericas = [v for v in variables if v not in categoricas]
    columnas = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else [])), numericas),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")), ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])

def candidatos():
    return {
        "clase mayoritaria (línea base)": armar(DummyClassifier(strategy="most_frequent")),
        "regresión logística": armar(LogisticRegression(max_iter=5000, class_weight="balanced")),
        "random forest": armar(RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_depth=8, class_weight="balanced_subsample",
                                                      random_state=SEMILLA, n_jobs=-1), escalar=False),
        "gradient boosting": armar(HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, class_weight="balanced",
                                                                  early_stopping=False, random_state=SEMILLA), escalar=False),
    }

validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMILLA)
filas = {}
for nombre, pipe in candidatos().items():
    r = cross_validate(pipe, X_train, y_train, cv=validacion, scoring={"f1": "f1_macro", "bal": "balanced_accuracy"})
    filas[nombre] = {"F1 macro cv": r["test_f1"].mean(), "desviación": r["test_f1"].std(), "exactitud balanceada cv": r["test_bal"].mean()}
comparacion = pd.DataFrame(filas).T.round(3)
comparacion""")
c.md("La desviación entre particiones es grande (0.06 a 0.09) y las diferencias entre los tres modelos reales son menores que eso. Se elige el de mayor F1 macro, pero es casi un empate.")
c.code("""ganador = comparacion.drop("clase mayoritaria (línea base)")["F1 macro cv"].idxmax()
print("Modelo elegido:", ganador)
modelo = candidatos()[ganador].fit(X_train, y_train)""")

c.md("""##### Evaluación en el conjunto de prueba

Como la etapa es ordinal, se agregan métricas que miden **qué tan lejos** se equivoca el modelo.""")
c.code("""ORDEN = [1, 2, 3, 4]

def ordinales(real, pred):
    y, p = np.asarray(real), np.asarray(pred)
    return {"exactitud": float((y == p).mean()), "F1 macro": f1_score(y, p, labels=ORDEN, average="macro", zero_division=0),
            "error medio (etapas)": float(np.abs(y - p).mean()), "a lo sumo 1 etapa de error": float((np.abs(y - p) <= 1).mean()),
            "kappa cuadrático": float(cohen_kappa_score(y, p, weights="quadratic"))}

pred = modelo.predict(X_test)
base = candidatos()["clase mayoritaria (línea base)"].fit(X_train, y_train).predict(X_test)
pd.DataFrame({"clase mayoritaria (línea base)": ordinales(y_test, base), ganador: ordinales(y_test, pred)}).T.round(3)""")
c.code("""sns.heatmap(confusion_matrix(y_test, pred, labels=ORDEN), annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN)
plt.xlabel("Etapa predicha")
plt.ylabel("Etapa real")
plt.title("Matriz de confusión (prueba)")
plt.show()""")
c.md("""El modelo supera a la línea base en F1 macro y en kappa, pero **la línea base gana en las dos métricas de distancia** (error medio y "a lo sumo 1 etapa de error"): al predecir siempre una etapa intermedia casi nunca se equivoca por mucho. Por eso estas métricas no deben leerse solas.

### Predicciones fuera de muestra del entrenamiento

La prueba tiene solo 56 pacientes, así que se usan también las predicciones de validación cruzada sobre los 220 del entrenamiento.""")
c.code("""oof = cross_val_predict(candidatos()[ganador], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
print({k: round(v, 3) for k, v in ordinales(y_train, oof).items()})
pd.DataFrame(classification_report(y_train, oof, labels=ORDEN, output_dict=True, zero_division=0)).T.loc[["1", "2", "3", "4"]].round(3)""")
c.md("La etapa 4 se reconoce mejor que las intermedias, y la 1 casi no se puede evaluar. Las etapas 2 y 3 se confunden entre sí.")

c.md("""### ¿Sirve el seguimiento posterior?

Se vuelve a la pregunta de `N_Days` y `Status`, con regresión logística y los mismos pliegues.""")
c.code("""def f1_cv(cols, datos=train):
    r = cross_val_score(armar(LogisticRegression(max_iter=5000, class_weight="balanced"), cols), datos[cols], y_train, cv=validacion, scoring="f1_macro")
    return r.mean(), r.std()

seguimiento = ["n_days", "status"]
filas = {"15 variables de la consulta inicial": f1_cv(VARIABLES), "15 + N_Days y Status": f1_cv(VARIABLES + seguimiento),
         "solo N_Days y Status": f1_cv(seguimiento)}
pd.DataFrame(filas, index=["F1 macro cv", "desviación"]).T.round(3)""")
c.md("""Agregarlos a las 15 variables **no mejora** el resultado, y solos dan un F1 apenas por encima de la línea base. No inflan la métrica, pero se excluyen por principio: no existen en la consulta.

### Qué variables usa realmente el modelo

Las medianas por etapa **describen los datos**, pero no dicen qué usa el modelo. Se mide quitando una variable a la vez.""")
c.code("""completa = f1_cv(VARIABLES)[0]
diferencia = {v: f1_cv([x for x in VARIABLES if x != v])[0] - completa for v in VARIABLES}
pd.Series(diferencia).sort_values().round(4).rename("cambio de F1 macro al quitarla")""")
c.md("""Ninguna variable aporta más de unas pocas centésimas de F1 al quitarla, y las diferencias son menores que la desviación entre particiones. El modelo reparte su información entre variables redundantes. Por eso **no se debe afirmar** que una variable en particular "es la que importa" solo porque se vea distinta entre etapas.""")

c.md("### Predicción para un paciente nuevo")
c.code("""paciente = pd.DataFrame([{"age": 55.0, "bilirubin": 1.8, "cholesterol": 320.0, "albumin": 3.5, "copper": 90.0, "alk_phos": 1800.0, "sgot": 110.0,
                          "tryglicerides": 120.0, "platelets": 240.0, "prothrombin": 10.8, "sex": "F", "ascites": "N", "hepatomegaly": "Y",
                          "spiders": "N", "edema": "N"}])
puntajes = {int(e): round(float(p), 3) for e, p in zip(modelo.classes_, modelo.predict_proba(paciente[VARIABLES])[0])}
print("Etapa más probable:", int(modelo.predict(paciente[VARIABLES])[0]))
print("Puntajes (no son probabilidades calibradas):", puntajes)""")

c.md("""---
# Conclusiones
---

## Resultados

Con signos clínicos y análisis de laboratorio se puede estimar la etapa por encima de la línea base, pero con un error considerable: el F1 macro está alrededor de 0.4 a 0.5 y el kappa cuadrático es moderado. Las etapas extremas se reconocen mejor que las intermedias, y la etapa 1 casi no se puede evaluar por tener muy pocos pacientes.

Hay varios hallazgos metodológicos. Primero, `N_Days` y `Status` son seguimiento posterior: se midió que no inflan la métrica, pero igual se excluyen por principio. Segundo, la línea base gana en las métricas de distancia, así que las métricas ordinales no deben leerse aisladas. Tercero, describir los datos no es medir lo que el modelo usa: ninguna variable aporta mucho sola.

Limitaciones: **no es un diagnóstico**. Son pocos pacientes (276 completos, 56 en la prueba), los datos vienen de un ensayo de 1974 a 1984 y los pacientes incompletos se dejaron fuera. Los puntajes no son probabilidades calibradas.

### Referencias

> Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.
> Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: Model for decision making. *Hepatology*, 10(1), 1-7.
> Ludwig, J., Dickson, E. R., & McDonald, G. S. (1978). Staging of chronic nonsuppurative destructive cholangitis (syndrome of primary biliary cirrhosis). *Virchows Archiv A*, 379(2), 103-112.""")

if __name__ == "__main__":
    print(c.guardar("modelo_07_cirrosis"))

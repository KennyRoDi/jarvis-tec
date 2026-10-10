"""Modelo 06 · Clasificación del estado hepático por hepatitis C (donante, hepatitis, fibrosis o cirrosis).

Ejecutar desde backend/:  python -m features.modelo_06_hepatitis.train
Dataset: https://www.kaggle.com/fedesoriano/hepatitis-c-dataset (615 personas; donantes de sangre y pacientes con hepatitis C)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
No es una herramienta de diagnóstico: es un ejercicio educativo.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, intervalo_bootstrap, metricas_clasificacion

CARPETA = Path(__file__).parent
OBJETIVO = "clase"
ORDEN = ["donante", "hepatitis", "fibrosis", "cirrosis"]  # de sano a más grave
ENFERMEDAD = ["hepatitis", "fibrosis", "cirrosis"]
CLASES = {"0=Blood Donor": "donante", "1=Hepatitis": "hepatitis", "2=Fibrosis": "fibrosis", "3=Cirrhosis": "cirrosis"}
LABORATORIO = ["alb", "alp", "alt", "ast", "bil", "che", "chol", "crea", "ggt", "prot"]
NUMERICAS = ["age"] + LABORATORIO
CATEGORICAS: list[str] = []
# 11 variables: edad y 10 análisis. Se excluye el sexo (sin aporte medible y es un dato sensible). El ALP SÍ entra: sus 18
# vacíos están todos en pacientes, pero se comprobó que su aporte no es ese artefacto (ver `experimento_alp_random_forest`).
VARIABLES = NUMERICAS + CATEGORICAS
SEMILLA = 42
CV = dict(scoring="f1_macro", n_jobs=-1, error_score="raise")
CONJUNTOS = {
    "12 (todas, con sexo)": VARIABLES + ["sex"],
    "11 (las elegidas: edad + 10 análisis)": VARIABLES,
    "10 (edad + 9 análisis, sin ALP)": [v for v in VARIABLES if v != "alp"],
    "10 (solo laboratorio, sin edad)": LABORATORIO,
    "2 (solo edad y sexo)": ["age", "sex"],
    "11 + indicador de ALP faltante (diagnóstico, no se usa)": VARIABLES + ["alp_faltante"],
}

# 1. Análisis del problema: clasificar el estado hepático (donante sano, hepatitis, fibrosis o cirrosis) a partir de
#    análisis de sangre. Clasificación multiclase supervisada, muy desbalanceada. Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo, descarta el índice y la clase ambigua "suspect Blood Donor" (7 filas) y normaliza nombres."""
    df = pd.read_csv(CARPETA / "dataset.csv").drop(columns="Unnamed: 0")
    df.columns = [c.lower() for c in df.columns]
    df = df[df["category"].isin(CLASES)].reset_index(drop=True)
    df[OBJETIVO] = df.pop("category").map(CLASES)
    df["alp_faltante"] = df["alp"].isna().astype(int)  # solo para cuantificar el artefacto; no es una variable del modelo
    return df


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Duplicados: {int(df.drop(columns=OBJETIVO).duplicated().sum())}")
    print("Clases:", df[OBJETIVO].value_counts().reindex(ORDEN).to_dict())
    nulos = df[VARIABLES + ["alp"]].isna().sum()
    print("\nNulos:", nulos[nulos > 0].to_dict())
    print("ALP faltante por clase:", df[df.alp.isna()][OBJETIVO].value_counts().reindex(ORDEN, fill_value=0).to_dict())
    print("CHOL faltante por clase:", df[df.chol.isna()][OBJETIVO].value_counts().reindex(ORDEN, fill_value=0).to_dict())
    print("Filas con los 10 análisis idénticos (casi duplicados; edad o sexo distintos):", int(df[LABORATORIO].duplicated().sum()))
    print("\nEdad mediana por clase:", df.groupby(OBJETIVO).age.median().reindex(ORDEN).to_dict())
    print("% hombres por clase:", df.groupby(OBJETIVO).sex.apply(lambda s: round((s == "m").mean(), 2)).reindex(ORDEN).to_dict())
    print("\nResumen del laboratorio:\n", df[LABORATORIO + ["alp"]].describe().round(1).T)


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    sns.countplot(data=df, x=OBJETIVO, order=ORDEN)
    plt.title("Distribución de las clases")
    guardar_figura(figuras, "clases")

    fig, ejes = plt.subplots(2, 4, figsize=(15, 7))
    for eje, variable in zip(ejes.ravel(), ["alb", "ast", "alt", "ggt", "bil", "che", "chol", "age"]):
        sns.boxplot(data=df, x=OBJETIVO, y=variable, order=ORDEN, ax=eje)
        eje.set_title(variable)
        eje.set_xlabel("")
        eje.tick_params(axis="x", rotation=25)
    guardar_figura(figuras, "laboratorio_por_clase")

    faltantes = df.assign(alp_vacio=df.alp.isna(), chol_vacio=df.chol.isna()).groupby(OBJETIVO)[["alp_vacio", "chol_vacio"]].mean().reindex(ORDEN)
    faltantes.plot.bar()
    plt.ylabel("Proporción con el valor vacío")
    plt.title("Valores faltantes por clase (informativos)")
    plt.xticks(rotation=25)
    guardar_figura(figuras, "faltantes_por_clase")

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(df[NUMERICAS + ["alp"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Correlación entre variables")
    guardar_figura(figuras, "correlacion")


def armar(estimador, variables: list[str] | None = None, escalar: bool = True) -> Pipeline:
    """4. Modelo: imputación (mediana) + escala de las numéricas, one-hot del sexo y estimador, en un Pipeline."""
    variables = VARIABLES if variables is None else variables
    categoricas = [v for v in variables if v == "sex"]  # solo en el experimento: el modelo no usa el sexo
    numericas = [v for v in variables if v != "sex"]
    columnas = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else [])), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Clases ponderadas para que las raras (17-24 casos) cuenten: los puntajes NO son probabilidades calibradas."""
    return {
        "clase mayoritaria (línea base)": armar(DummyClassifier(strategy="most_frequent")),
        "regresión logística": armar(LogisticRegression(max_iter=5000, class_weight="balanced")),
        "k vecinos": armar(KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "random forest": armar(RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_depth=8,
                                                      class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=-1), escalar=False),
    }


def dividir(df: pd.DataFrame):
    """Partición 80/20 **estratificada** (con 21-30 casos por clase rara, cada uno cuenta)."""
    return train_test_split(df, df[OBJETIVO], test_size=0.2, stratify=df[OBJETIVO], random_state=SEMILLA)


def sensibilidad_y_especificidad(y_real, y_pred) -> dict:
    """Vista binaria: ¿enfermedad (hepatitis, fibrosis o cirrosis) o donante?"""
    real, pred = np.isin(np.asarray(y_real), ENFERMEDAD), np.isin(np.asarray(y_pred), ENFERMEDAD)
    return {"sensibilidad": round(float(pred[real].mean()), 4), "especificidad": round(float((~pred[~real]).mean()), 4)}


def por_clase(y_real, y_pred) -> dict:
    informe = classification_report(y_real, y_pred, labels=ORDEN, output_dict=True, zero_division=0)
    return {c: {k: round(float(v), 3) for k, v in informe[c].items()} for c in ORDEN}


def f1_macro_de(y, pred) -> float:
    return float(f1_score(y, pred, labels=ORDEN, average="macro", zero_division=0))


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba (122 personas: solo 4 a 6 por cada clase rara)."""
    pred = pipeline.predict(X_test)
    metricas = metricas_clasificacion(y_test, pred, orden=ORDEN)
    metricas["por_clase"] = por_clase(y_test, pred)
    metricas["enfermedad_vs_donante"] = sensibilidad_y_especificidad(y_test, pred)
    # Con 4 a 6 casos por clase la prueba es muy ruidosa: intervalos de confianza del 95 % por bootstrap.
    metricas["ic95"] = {
        "f1_macro": intervalo_bootstrap(np.asarray(y_test), pred, f1_macro_de),
        "accuracy": intervalo_bootstrap(np.asarray(y_test), pred, lambda y, p: float((y == p).mean())),
    }
    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas
    sns.heatmap(confusion_matrix(y_test, pred, labels=ORDEN), annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión (prueba)")
    guardar_figura(figuras, "matriz_confusion")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    train, test, y_train, y_test = dividir(df)
    X_train, X_test = train[VARIABLES], test[VARIABLES]

    # La selección usa solo el entrenamiento (validación cruzada estratificada repetida) con el F1 macro.
    validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        f1 = cross_val_score(pipe, X_train, y_train, cv=validacion, **CV)
        bal = cross_val_score(pipe, X_train, y_train, cv=validacion, **{**CV, "scoring": "balanced_accuracy"})
        comparacion[nombre] = {"cv_f1_macro": round(float(f1.mean()), 4), "cv_f1_desv": round(float(f1.std()), 4), "cv_balanced_accuracy": round(float(bal.mean()), 4)}
        log(f"{nombre:32s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}   exactitud balanceada cv = {bal.mean():.3f}")
    ganador = max((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_f1_macro"])
    log(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["clase mayoritaria (línea base)"].fit(X_train, y_train)
    metricas["baseline"] = {k: v for k, v in metricas_clasificacion(y_test, base.predict(X_test), orden=ORDEN).items() if k != "matriz_confusion"}
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)
    metricas["casos_por_clase_prueba"] = {c: int((y_test == c).sum()) for c in ORDEN}

    # Predicciones fuera de muestra del entrenamiento (486 personas): estimaciones por clase menos ruidosas que la prueba.
    oof = cross_val_predict(candidatos()[ganador], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
    metricas["oof_entrenamiento"] = {**{k: v for k, v in metricas_clasificacion(y_train, oof, orden=ORDEN).items() if k in ("f1_macro", "accuracy", "recall_macro")},
                                      "por_clase": por_clase(y_train, oof), "enfermedad_vs_donante": sensibilidad_y_especificidad(y_train, oof)}

    # Experimento: ¿qué variables hacen falta? (regresión logística, F1 macro de validación cruzada, solo entrenamiento)
    experimento = {}
    for nombre, cols in CONJUNTOS.items():
        f1 = cross_val_score(armar(LogisticRegression(max_iter=5000, class_weight="balanced"), cols), train[cols], y_train, cv=validacion, **CV)
        experimento[nombre] = {"variables": len(cols), "cv_f1_macro": round(float(f1.mean()), 4), "cv_f1_desv": round(float(f1.std()), 4)}
        log(f"  {nombre:56s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}")
    metricas["experimento_variables"] = experimento

    # ¿El aporte del ALP es un artefacto de sus vacíos (que están todos en pacientes) o señal real? Se compara con (1) quitarlo,
    # (2) rellenar los vacíos con valores observados al azar (sin pista de faltante) y (3) usar solo el indicador de faltante.
    # Si fuera un artefacto, (2) lo haría desaparecer y (3) lo reproduciría; ocurre lo contrario.
    parametros = candidatos()["random forest"].named_steps["modelo"].get_params()
    azar = train.copy()
    vacios = azar["alp"].isna()
    azar.loc[vacios, "alp"] = np.random.default_rng(SEMILLA).choice(train["alp"].dropna().to_numpy(), int(vacios.sum()))
    rf = lambda: RandomForestClassifier(**parametros)  # noqa: E731
    casos = {
        "random forest con ALP (elegido)": (train, VARIABLES),
        "random forest sin ALP": (train, [v for v in VARIABLES if v != "alp"]),
        "random forest con los vacíos de ALP rellenados al azar con valores observados": (azar, VARIABLES),
        "random forest solo con el indicador de ALP faltante (sin ALP)": (train, [v for v in VARIABLES if v != "alp"] + ["alp_faltante"]),
        "random forest solo en las filas con ALP observada, con ALP": (train[train.alp.notna()], VARIABLES),
        "random forest solo en las filas con ALP observada, sin ALP": (train[train.alp.notna()], [v for v in VARIABLES if v != "alp"]),
    }
    experimento_alp = {}
    for nombre, (datos, cols) in casos.items():
        f1 = cross_val_score(armar(rf(), cols, escalar=False), datos[cols], datos[OBJETIVO], cv=validacion, **CV)
        experimento_alp[nombre] = {"cv_f1_macro": round(float(f1.mean()), 4), "cv_f1_desv": round(float(f1.std()), 4)}
        log(f"  {nombre:78s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}")
    metricas["experimento_alp_random_forest"] = experimento_alp

    # La alternativa (regresión logística) rinde casi igual en validación cruzada pero con otro compromiso sensibilidad/especificidad.
    oof_lr = cross_val_predict(candidatos()["regresión logística"], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
    metricas["oof_regresion_logistica"] = {"f1_macro": round(f1_macro_de(y_train, oof_lr), 4), "recall_hepatitis": por_clase(y_train, oof_lr)["hepatitis"]["recall"],
                                           "enfermedad_vs_donante": sensibilidad_y_especificidad(y_train, oof_lr)}

    return {
        "pipeline": pipeline, "metricas": metricas,
        "rango": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAS},  # rango visto en entrenamiento
        "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0],
    }


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"], variables=VARIABLES, rango=r["rango"])
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k not in ("comparacion_cv", "experimento_variables", "experimento_alp_random_forest", "por_clase", "oof_entrenamiento")})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

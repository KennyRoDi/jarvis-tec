"""Modelo 03 · Clasificación de la calidad del vino.

Ejecutar desde backend/:  python -m features.modelo_03_vino.train
Dataset: https://www.kaggle.com/rajyellow46/wine-quality (vinho verde tinto y blanco, Cortez et al., 2009)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, metricas_clasificacion

CARPETA = Path(__file__).parent
OBJETIVO = "clase"
ORDEN = ["baja", "media", "alta"]  # orden natural de la calidad (para tablas y matrices)
NUMERICAS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol",
]
CATEGORICAS = ["tipo"]
VARIABLES = NUMERICAS + CATEGORICAS
SEMILLA = 42

# 1. Análisis del problema: clasificar el vino en calidad baja, media o alta a partir de sus propiedades
#    fisicoquímicas. Clasificación multiclase supervisada. Ver analisis.md.


def agrupar_calidad(puntaje: pd.Series) -> pd.Series:
    """La puntuación sensorial (3-9) se agrupa: las clases extremas casi no tienen ejemplos."""
    return pd.Series(np.where(puntaje <= 5, "baja", np.where(puntaje == 6, "media", "alta")), index=puntaje.index)


def cargar_datos(quitar_duplicados: bool = True) -> pd.DataFrame:
    """Lee el CSV crudo, normaliza nombres y (por defecto) descarta las filas duplicadas antes de dividir."""
    df = pd.read_csv(CARPETA / "dataset.csv")
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    df = df.rename(columns={"type": "tipo"})
    if quitar_duplicados:
        df = df.drop_duplicates().reset_index(drop=True)
    df[OBJETIVO] = agrupar_calidad(df["quality"])
    return df


def dividir(df: pd.DataFrame):
    """Partición 80/20 **estratificada**: conserva la proporción de las tres clases en entrenamiento y prueba."""
    X, y = df[VARIABLES], df[OBJETIVO]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)


def entender(df: pd.DataFrame, n_crudo: int) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas crudas: {n_crudo}  Filas sin duplicados: {len(df)} ({n_crudo - len(df)} duplicadas descartadas)")
    print("\nTipos:\n", df[VARIABLES].dtypes)
    nulos = df[VARIABLES].isna().sum()
    print("\nNulos por columna:", nulos[nulos > 0].to_dict(), "| filas con algún nulo:", int(df[VARIABLES].isna().any(axis=1).sum()))
    print("\nPuntuación (quality):", df["quality"].value_counts().sort_index().to_dict())
    print("Clases:", df[OBJETIVO].value_counts(normalize=True).round(3).reindex(ORDEN).to_dict())
    print("\nResumen estadístico:\n", df[NUMERICAS].describe().round(3).T)


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    sns.countplot(data=df, x=OBJETIVO, hue="tipo", order=ORDEN)
    plt.title("Clases de calidad por tipo de vino")
    guardar_figura(figuras, "clases")

    fig, ejes = plt.subplots(1, 3, figsize=(12, 4))
    for eje, variable in zip(ejes, ["alcohol", "volatile_acidity", "density"]):
        sns.boxplot(data=df, x=OBJETIVO, y=variable, order=ORDEN, ax=eje)
        eje.set_title(variable)
    guardar_figura(figuras, "variables_por_clase")

    corr = df[NUMERICAS + ["quality"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Correlación entre variables")
    guardar_figura(figuras, "correlacion")
    print("\nCorrelación con la puntuación:\n", corr["quality"].drop("quality").sort_values().round(2).to_string())
    print("\nClase por tipo:\n", pd.crosstab(df["tipo"], df[OBJETIVO], normalize="index").round(2).reindex(columns=ORDEN))


def armar(estimador, escalar: bool = True) -> Pipeline:
    """4. Modelo: imputación de nulos (mediana) + escala + codificación del tipo + estimador, en un Pipeline."""
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else []))
    columnas = ColumnTransformer([
        ("num", numericas, NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    return {
        "mayoritaria (línea base)": armar(DummyClassifier(strategy="most_frequent")),
        "regresión logística": armar(LogisticRegression(max_iter=3000, class_weight="balanced")),
        "svm rbf": armar(SVC(C=3, class_weight="balanced", probability=True, random_state=SEMILLA)),
        "random forest": armar(RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, max_depth=16, class_weight="balanced_subsample",  # compacto: ~3 MB en git
            random_state=SEMILLA, n_jobs=-1), escalar=False),
        "gradient boosting": armar(HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, class_weight="balanced", early_stopping=False,
            random_state=SEMILLA), escalar=False),
    }


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path) -> dict:
    """5. Evaluación sobre el conjunto de prueba."""
    y_pred = pipeline.predict(X_test)
    metricas = metricas_clasificacion(y_test, y_pred, orden=ORDEN)  # matriz: filas = real, columnas = predicho
    matriz = confusion_matrix(y_test, y_pred, labels=ORDEN)
    metricas["por_clase"] = {c: {k: round(float(v), 3) for k, v in d.items()}
                             for c, d in classification_report(y_test, y_pred, labels=ORDEN, output_dict=True, zero_division=0).items()
                             if c in ORDEN}
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión (prueba)")
    guardar_figura(figuras, "matriz_confusion")
    return metricas


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    crudo = cargar_datos(quitar_duplicados=False)
    df = cargar_datos()
    entender(df, len(crudo))
    explorar(df, figuras)

    X_train, X_test, y_train, y_test = dividir(df)

    # La selección usa solo el entrenamiento (validación cruzada estratificada repetida); la prueba, una vez.
    validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        f1 = cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="f1_macro", n_jobs=-1)
        exactitud = cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="accuracy", n_jobs=-1)
        comparacion[nombre] = {"cv_f1_macro": round(float(f1.mean()), 3), "cv_f1_desv": round(float(f1.std()), 3),
                               "cv_accuracy": round(float(exactitud.mean()), 3)}
        print(f"{nombre:26s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}   accuracy cv = {exactitud.mean():.3f}")

    ganador = max((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_f1_macro"])
    print(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["mayoritaria (línea base)"].fit(X_train, y_train)
    metricas["baseline"] = {k: v for k, v in metricas_clasificacion(y_test, base.predict(X_test)).items() if k != "matriz_confusion"}
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)

    # Experimento: por qué se descartan los duplicados. Con ellos, filas idénticas caen en entrenamiento y
    # prueba y la métrica sube sin que el modelo generalice mejor.
    Xc, yc = crudo[VARIABLES], crudo[OBJETIVO]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=SEMILLA)
    inflado = clone(candidatos()[ganador]).fit(Xc_tr, yc_tr)
    met_inf = metricas_clasificacion(yc_te, inflado.predict(Xc_te))
    copias = Xc_te.merge(Xc_tr.drop_duplicates(), on=VARIABLES, how="left", indicator=True)["_merge"].eq("both")
    metricas["experimento_duplicados"] = {
        "accuracy_con_duplicados": met_inf["accuracy"], "f1_macro_con_duplicados": met_inf["f1_macro"],
        "filas_duplicadas": int(len(crudo) - len(df)), "n_prueba_con_duplicados": len(Xc_te),
        "prueba_con_copia_en_entrenamiento": int(copias.sum()),
    }

    # Rango visto en entrenamiento: la API avisa cuando una medida lo excede (core.modelos.fuera_de_rango).
    rango = {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAS}
    guardar_modelo(CARPETA, pipeline, metricas, entrada_ejemplo=X_test.iloc[[0]].to_dict("records")[0],
                   variables=VARIABLES, rango=rango)
    print("\nMétricas:", {k: v for k, v in metricas.items() if k not in ("comparacion_cv", "matriz_confusion", "por_clase")})
    print("Por clase:", metricas["por_clase"])
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

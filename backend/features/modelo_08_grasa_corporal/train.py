"""Modelo 08 · Predicción del porcentaje de grasa corporal.

Ejecutar desde backend/:  python -m features.modelo_08_grasa_corporal.train
Dataset: https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset (252 hombres, Penrose et al., 1985)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, metricas_regresion

CARPETA = Path(__file__).parent
OBJETIVO = "bodyfat"
# Columnas del modelo (y campos de Entrada en router.py). Unidades métricas: el dataset trae lb y pulgadas.
VARIABLES = [
    "age", "weight_kg", "height_cm", "neck_cm", "chest_cm", "abdomen_cm", "hip_cm",
    "thigh_cm", "knee_cm", "ankle_cm", "biceps_cm", "forearm_cm", "wrist_cm",
]
SEMILLA = 42
LB_A_KG = 0.45359237
PULGADA_A_CM = 2.54

# 1. Análisis del problema: estimar el porcentaje de grasa corporal de un hombre adulto con medidas
#    corporales simples (en lugar del pesaje hidrostático). Regresión supervisada. Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo, pasa a unidades métricas y nombres snake_case. Conserva `density` solo para el experimento."""
    df = pd.read_csv(CARPETA / "dataset.csv")
    df.columns = [c.lower() for c in df.columns]
    df["weight_kg"] = (df.pop("weight") * LB_A_KG).round(2)
    df["height_cm"] = (df.pop("height") * PULGADA_A_CM).round(1)
    return df.rename(columns={c: f"{c}_cm" for c in
                              ["neck", "chest", "abdomen", "hip", "thigh", "knee", "ankle", "biceps", "forearm", "wrist"]})


def limpiar(df: pd.DataFrame, imprimir: bool = True) -> pd.DataFrame:
    """Descarta solo los registros físicamente imposibles (ver analisis.md, etapa 3)."""
    imposibles = (df[OBJETIVO] <= 0) | (df["height_cm"] < 120)  # grasa 0 % y estatura de 75 cm
    if imprimir:
        print(f"Registros imposibles descartados: {int(imposibles.sum())} -> filas {df.index[imposibles].tolist()}")
    return df[~imposibles].reset_index(drop=True)


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:", int(df.isna().sum().sum()))
    print("\nResumen estadístico:\n", df.describe().round(2).T)


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    sns.histplot(df[OBJETIVO], kde=True)
    plt.xlabel("Grasa corporal (%)")
    plt.title("Distribución del porcentaje de grasa corporal")
    guardar_figura(figuras, "distribucion_objetivo")

    corr = df[VARIABLES + [OBJETIVO]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Correlación entre variables")
    guardar_figura(figuras, "correlacion")

    sns.regplot(data=df, x="abdomen_cm", y=OBJETIVO, scatter_kws={"alpha": 0.6}, seed=SEMILLA)  # semilla: el IC es por bootstrap
    plt.title("Circunferencia abdominal vs. grasa corporal")
    guardar_figura(figuras, "abdomen_vs_grasa")

    print("\nCorrelación con el objetivo:\n", corr[OBJETIVO].drop(OBJETIVO).sort_values().round(2).to_string())


def candidatos() -> dict:
    """4. Modelo: cada candidato es un Pipeline completo (mismo preprocesamiento en entrenamiento y API)."""
    def con_escala(estimador):
        return Pipeline([("escala", StandardScaler()), ("modelo", estimador)])

    return {
        "media (línea base)": Pipeline([("modelo", DummyRegressor(strategy="mean"))]),
        "regresión lineal": con_escala(LinearRegression()),
        "ridge": con_escala(RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "lasso": con_escala(LassoCV(cv=5, random_state=SEMILLA, max_iter=20000)),
        "random forest": Pipeline([("modelo", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEMILLA, n_jobs=-1))]),
    }


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba."""
    y_pred = pipeline.predict(X_test)
    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas_regresion(y_test, y_pred)
    plt.scatter(y_test, y_pred, alpha=0.7)
    limite = max(y_test.max(), y_pred.max())
    plt.plot([0, limite], [0, limite], "r--")
    plt.xlabel("Grasa corporal real (%)")
    plt.ylabel("Grasa corporal predicha (%)")
    plt.title("Real vs. predicho (prueba)")
    guardar_figura(figuras, "real_vs_predicho")
    return metricas_regresion(y_test, y_pred)


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    X, y = df[VARIABLES], df[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

    # La selección se hace solo con el conjunto de entrenamiento (validación cruzada repetida);
    # el conjunto de prueba se usa una única vez, con el modelo elegido.
    validacion = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        rmse = -cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="neg_root_mean_squared_error")
        r2 = cross_val_score(pipe, X_train, y_train, cv=validacion, scoring="r2")
        comparacion[nombre] = {"cv_rmse": round(float(rmse.mean()), 3), "cv_rmse_desv": round(float(rmse.std()), 3),
                               "cv_r2": round(float(r2.mean()), 3)}
        log(f"{nombre:20s} RMSE cv = {rmse.mean():.3f} ± {rmse.std():.3f}   R² cv = {r2.mean():.3f}")

    ganador = min((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_rmse"])
    log(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    metricas["baseline_media"] = metricas_regresion(y_test, candidatos()["media (línea base)"].fit(X_train, y_train).predict(X_test))
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)

    # Experimento: por qué `density` está excluida (BodyFat se calcula a partir de ella con la ecuación de Siri).
    con_densidad = df[VARIABLES + ["density"]]
    Xd_train, Xd_test = con_densidad.loc[X_train.index], con_densidad.loc[X_test.index]
    fuga = candidatos()["regresión lineal"].fit(Xd_train, y_train)
    metricas["r2_con_density_fuga"] = metricas_regresion(y_test, fuga.predict(Xd_test))["r2"]

    return {
        "pipeline": pipeline, "metricas": metricas,
        # Rango visto en entrenamiento: los modelos lineales extrapolan mal fuera de él (ver analisis.md) y el router avisa.
        "rango": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in VARIABLES},
        "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0],
    }


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = limpiar(cargar_datos())
    entender(df)
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"], variables=VARIABLES, rango=r["rango"])
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k != "comparacion_cv"})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

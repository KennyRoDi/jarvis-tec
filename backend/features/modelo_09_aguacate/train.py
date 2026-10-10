"""Modelo 09 · Predicción del precio del aguacate.

Ejecutar desde backend/:  python -m features.modelo_09_aguacate.train
Dataset: https://www.kaggle.com/neuromusic/avocado-prices (Hass Avocado Board; 54 regiones × 2 tipos × 169 semanas)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, metricas_regresion
from features.modelo_09_aguacate.preprocesamiento import CaracteristicasFecha

CARPETA = Path(__file__).parent
OBJETIVO = "precio"
VARIABLES = ["region", "tipo", "fecha"]  # lo que recibe el pipeline (y la API)
SEMILLA = 42
FRACCION_PRUEBA = 0.20  # últimas semanas: se evalúa prediciendo el futuro, no filas al azar
SEMANAS_CV = 20

# 1. Análisis del problema: estimar el precio promedio de un aguacate Hass según región, tipo (convencional u
#    orgánico) y fecha. Regresión supervisada sobre una serie de tiempo panel. Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo. Descarta la columna de índice y los volúmenes (ver analisis.md, etapa 2)."""
    df = pd.read_csv(CARPETA / "dataset.csv", parse_dates=["Date"])
    df = df.rename(columns={"Date": "fecha", "AveragePrice": OBJETIVO, "type": "tipo"})
    return df[["fecha", "region", "tipo", OBJETIVO]].sort_values(["fecha", "region", "tipo"]).reset_index(drop=True)


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Semanas: {df.fecha.nunique()}  Regiones: {df.region.nunique()}")
    print(f"Rango: {df.fecha.min().date()} a {df.fecha.max().date()}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos:", int(df.isna().sum().sum()), "| duplicados (fecha, región, tipo):", int(df.duplicated(["fecha", "region", "tipo"]).sum()))
    print("\nResumen del precio:\n", df[OBJETIVO].describe().round(2))


def explorar(df: pd.DataFrame, figuras: Path, fecha_corte: pd.Timestamp) -> None:
    """3. Exploración de los datos."""
    sns.histplot(df[OBJETIVO], kde=True)
    plt.xlabel("Precio promedio por aguacate (USD)")
    plt.title("Distribución del precio")
    guardar_figura(figuras, "distribucion_objetivo")

    por_mes = df.assign(mes=df.fecha.dt.month).groupby(["mes", "tipo"])[OBJETIVO].mean().reset_index()
    sns.lineplot(data=por_mes, x="mes", y=OBJETIVO, hue="tipo", marker="o")
    plt.title("Estacionalidad: precio medio por mes")
    plt.xticks(range(1, 13))
    guardar_figura(figuras, "estacionalidad")

    nacional = df[df.region == "TotalUS"]
    sns.lineplot(data=nacional, x="fecha", y=OBJETIVO, hue="tipo")
    plt.axvline(fecha_corte, color="red", linestyle="--", label="inicio de la prueba")
    plt.legend()
    plt.title("Precio nacional (TotalUS) a lo largo del tiempo")
    plt.xticks(rotation=30)
    guardar_figura(figuras, "serie_nacional")

    medias = df.groupby("region")[OBJETIVO].mean().sort_values()
    extremos = pd.concat([medias.head(8), medias.tail(8)])
    extremos.plot.barh(color=["#2a9d8f"] * 8 + ["#e76f51"] * 8)
    plt.xlabel("Precio medio (USD)")
    plt.title("Regiones más baratas y más caras")
    guardar_figura(figuras, "regiones_extremas")

    print("\nPrecio medio por tipo:", df.groupby("tipo")[OBJETIVO].mean().round(2).to_dict())
    print("Precio medio por mes:", df.groupby(df.fecha.dt.month)[OBJETIVO].mean().round(2).to_dict())


def armar(estimador, categoricas: list[str], numericas: list[str], escala="passthrough") -> Pipeline:
    """4. Modelo: (fecha -> mes/semana/t) + codificación + estimador, todo en un único Pipeline."""
    columnas = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ("num", escala if numericas else "drop", numericas),
    ])
    return Pipeline([("fecha", CaracteristicasFecha()), ("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Se comparan variantes con y sin la tendencia `t`: la evidencia queda en metricas.json (ver analisis.md)."""
    grupo = ["region", "tipo"]
    ridge = lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))  # noqa: E731
    bosque = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_depth=14,  # noqa: E731
                                           random_state=SEMILLA, n_jobs=-1)
    boosting = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=SEMILLA)  # noqa: E731
    return {
        "efecto región y tipo (línea base)": armar(LinearRegression(), grupo, []),
        "ridge estacional": armar(ridge(), grupo + ["mes"], []),
        "ridge estacional + tendencia": armar(ridge(), grupo + ["mes"], ["t"], StandardScaler()),
        "random forest": armar(bosque(), grupo, ["mes", "semana"]),
        "random forest + tendencia": armar(bosque(), grupo, ["t", "mes", "semana"]),
        "gradient boosting": armar(boosting(), grupo, ["mes", "semana"]),
        "gradient boosting + tendencia": armar(boosting(), grupo, ["t", "mes", "semana"]),
    }


def particion_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Entrenamiento = semanas anteriores; prueba = las últimas FRACCION_PRUEBA de las semanas."""
    fechas = np.sort(df.fecha.unique())
    corte = pd.Timestamp(fechas[int(len(fechas) * (1 - FRACCION_PRUEBA))])
    return df[df.fecha < corte].reset_index(drop=True), df[df.fecha >= corte].reset_index(drop=True), corte


def cv_por_semanas(df_train: pd.DataFrame, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación cruzada con ventana creciente por **semanas** (todas las filas de una semana van juntas)."""
    fechas = np.sort(df_train.fecha.unique())
    pliegues = []
    for entrena, prueba in TimeSeriesSplit(n_splits=n_splits, test_size=SEMANAS_CV).split(fechas):
        pliegues.append((np.flatnonzero(df_train.fecha.isin(fechas[entrena])),
                         np.flatnonzero(df_train.fecha.isin(fechas[prueba]))))
    return pliegues


def evaluar(pipeline: Pipeline, test: pd.DataFrame, figuras: Path) -> dict:
    """5. Evaluación sobre las últimas semanas (futuro respecto del entrenamiento)."""
    y_pred = pipeline.predict(test[VARIABLES])
    plt.scatter(test[OBJETIVO], y_pred, alpha=0.3, s=10)
    limite = max(test[OBJETIVO].max(), y_pred.max())
    plt.plot([0, limite], [0, limite], "r--")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.title("Real vs. predicho (últimas semanas)")
    guardar_figura(figuras, "real_vs_predicho")

    nacional = test.assign(prediccion=y_pred)
    nacional = nacional[nacional.region == "TotalUS"]
    for tipo, color in [("conventional", "tab:blue"), ("organic", "tab:orange")]:
        parte = nacional[nacional.tipo == tipo]
        plt.plot(parte.fecha, parte[OBJETIVO], color=color, label=f"{tipo} real")
        plt.plot(parte.fecha, parte.prediccion, color=color, linestyle="--", label=f"{tipo} predicho")
    plt.legend()
    plt.xticks(rotation=30)
    plt.title("Prueba: precio nacional real vs. predicho")
    guardar_figura(figuras, "prueba_nacional")
    return metricas_regresion(test[OBJETIVO], y_pred)


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    train, test, corte = particion_temporal(df)
    explorar(df, figuras, corte)
    print(f"\nEntrenamiento: {len(train)} filas hasta {train.fecha.max().date()} | prueba: {len(test)} filas desde {corte.date()}")

    # La selección usa solo el entrenamiento (validación cruzada con ventana creciente por semanas).
    pliegues = cv_por_semanas(train)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        rmse = -cross_val_score(pipe, train[VARIABLES], train[OBJETIVO], cv=pliegues, scoring="neg_root_mean_squared_error")
        r2 = cross_val_score(pipe, train[VARIABLES], train[OBJETIVO], cv=pliegues, scoring="r2")
        comparacion[nombre] = {"cv_rmse": round(float(rmse.mean()), 3), "cv_rmse_desv": round(float(rmse.std()), 3),
                               "cv_r2": round(float(r2.mean()), 3)}
        print(f"{nombre:36s} RMSE cv = {rmse.mean():.3f} ± {rmse.std():.3f}   R² cv = {r2.mean():.3f}")

    ganador = min((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_rmse"])
    print(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(train[VARIABLES], train[OBJETIVO])

    metricas = evaluar(pipeline, test, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["efecto región y tipo (línea base)"].fit(train[VARIABLES], train[OBJETIVO])
    metricas["baseline"] = metricas_regresion(test[OBJETIVO], base.predict(test[VARIABLES]))
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(train), len(test)

    # Despliegue: las métricas son las de la partición temporal, pero el modelo que sirve la API se reentrena
    # con TODAS las semanas para que conozca los precios más recientes (hasta la última fecha del dataset).
    final = candidatos()[ganador].fit(df[VARIABLES], df[OBJETIVO])
    guardar_modelo(CARPETA, final, metricas,
                   entrada_ejemplo={"region": "TotalUS", "tipo": "conventional", "fecha": str(df.fecha.max().date())},
                   variables=VARIABLES, fecha_max=str(df.fecha.max().date()), reentrenado_con_todo=True)
    print("\nMétricas:", {k: v for k, v in metricas.items() if k != "comparacion_cv"})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

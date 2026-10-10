"""Modelo 02 · Predicción del precio de un automóvil usado.

Ejecutar desde backend/:  python -m features.modelo_02_autos.train
Dataset: https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv (301 autos usados; CarDekho)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, intervalo_bootstrap, metricas_regresion
from features.modelo_02_autos.razon import RazonAlPrecioActual

CARPETA = Path(__file__).parent
OBJETIVO = "selling_price"
NUMERICAS = ["year", "present_price", "kms_driven", "owner"]
CATEGORICAS = ["fuel_type", "seller_type", "transmission"]
VARIABLES = NUMERICAS + CATEGORICAS
CON_RANGO = ["year", "present_price", "kms_driven"]  # la API avisa si una de estas queda fuera de lo visto en el entrenamiento
SEMILLA = 42
PRECIO_ALTO = 15  # experimento de extrapolación: se entrena con present_price <= 15 y se prueba con los más caros

# 1. Análisis del problema: estimar el precio de reventa (lakhs INR) a partir de las características del vehículo.
#    Problema de regresión supervisada con pocos datos (301 filas). Ver analisis.md.


def cargar_datos(quitar_duplicados: bool = True) -> pd.DataFrame:
    """Lee el CSV crudo, descarta `car_name` y (por defecto) las filas duplicadas antes de dividir."""
    df = pd.read_csv(CARPETA / "dataset.csv")
    df.columns = [c.lower() for c in df.columns]
    df = df.drop(columns=["car_name"])  # ~100 nombres distintos en 301 filas: no generaliza
    return df.drop_duplicates().reset_index(drop=True) if quitar_duplicados else df


def entender(df: pd.DataFrame, n_crudo: int) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas crudas: {n_crudo}  Filas sin duplicados: {len(df)} ({n_crudo - len(df)} duplicadas descartadas)  Columnas: {df.shape[1]}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:", int(df.isna().sum().sum()))
    print("\nResumen estadístico:\n", df.describe().round(2).T)
    print("\nCombustible:", df.fuel_type.value_counts().to_dict(), "| vendedor:", df.seller_type.value_counts().to_dict())
    print("Transmisión:", df.transmission.value_counts().to_dict(), "| dueños anteriores:", df.owner.value_counts().sort_index().to_dict())
    razon = df[OBJETIVO] / df.present_price
    print(f"\nRazón reventa / precio de agencia: mínimo {razon.min():.2f}, mediana {razon.median():.2f}, máximo {razon.max():.2f}")


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    sns.histplot(df[OBJETIVO], kde=True)
    plt.title("Distribución del precio de venta")
    guardar_figura(figuras, "distribucion_objetivo")

    sns.heatmap(df[NUMERICAS + [OBJETIVO]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlación entre variables numéricas")
    guardar_figura(figuras, "correlacion")

    sns.scatterplot(data=df, x="present_price", y=OBJETIVO, hue="fuel_type")
    plt.title("Precio actual vs. precio de venta")
    guardar_figura(figuras, "present_vs_selling")

    razones = (df[OBJETIVO] / df.present_price).groupby(df.year)
    plt.boxplot([g.to_numpy() for _, g in razones], tick_labels=[str(a) for a, _ in razones])
    plt.ylabel("Reventa / precio de agencia")
    plt.xticks(rotation=45)
    plt.title("La razón de reventa baja con la antigüedad del auto")
    guardar_figura(figuras, "razon_por_anio")

    razon = df[OBJETIVO] / df.present_price
    print("\nCorrelación con el precio de venta:", df[NUMERICAS].corrwith(df[OBJETIVO]).round(2).to_dict())
    print("Razón mediana por año:", razon.groupby(df.year).median().round(2).to_dict())
    print("Mediana de reventa y de agencia por combustible:", df.groupby("fuel_type")[[OBJETIVO, "present_price"]].median().round(2).to_dict("index"))
    print("Autos con precio de agencia mayor a 30:", int((df.present_price > 30).sum()), "| kms mayores a 200 000:", int((df.kms_driven > 200_000).sum()))


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUMERICAS), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])


def armar(estimador) -> Pipeline:
    """4. Modelo: preprocesamiento + estimador en un único Pipeline (lo mismo se usa en la API)."""
    return Pipeline([("preprocesador", preprocesador()), ("modelo", estimador)])


def bosque() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, random_state=SEMILLA, n_jobs=-1)


def boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=SEMILLA)


def candidatos() -> dict:
    """Las dos primeras son líneas base (no se pueden elegir). Los modelos "(razón)" aprenden la razón reventa / precio de agencia."""
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


def es_linea_base(nombre: str) -> bool:
    return "línea base" in nombre


def dividir(df: pd.DataFrame):
    """Partición 80/20 con semilla fija."""
    return train_test_split(df[VARIABLES], df[OBJETIVO], test_size=0.2, random_state=SEMILLA)


def evaluar(pipeline, X_test, y_test, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba (60 autos: los intervalos de confianza son amplios)."""
    y_pred = pipeline.predict(X_test)
    metricas = metricas_regresion(y_test, y_pred)
    y, p = np.asarray(y_test), np.asarray(y_pred)
    metricas["ic95"] = {"rmse": intervalo_bootstrap(y, p, lambda a, b: float(np.sqrt(mean_squared_error(a, b)))),
                        "mae": intervalo_bootstrap(y, p, lambda a, b: float(mean_absolute_error(a, b))),
                        "r2": intervalo_bootstrap(y, p, lambda a, b: float(r2_score(a, b)))}
    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas
    plt.scatter(y_test, y_pred, alpha=0.7)
    limite = max(y_test.max(), y_pred.max())
    plt.plot([0, limite], [0, limite], "r--")
    plt.xlabel("Precio real (lakhs INR)")
    plt.ylabel("Precio predicho (lakhs INR)")
    plt.title("Real vs. predicho (prueba)")
    guardar_figura(figuras, "real_vs_predicho")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    X_train, X_test, y_train, y_test = dividir(df)

    # La selección usa solo el entrenamiento (validación cruzada repetida, error en lakhs); la prueba se usa una vez.
    validacion = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        r = cross_validate(pipe, X_train, y_train, cv=validacion, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        comparacion[nombre] = {"cv_rmse": round(float(-r["test_rmse"].mean()), 3), "cv_rmse_desv": round(float(r["test_rmse"].std()), 3),
                               "cv_r2": round(float(r["test_r2"].mean()), 3)}
        log(f"{nombre:36s} RMSE cv = {-r['test_rmse'].mean():.3f} ± {r['test_rmse'].std():.3f}   R² cv = {r['test_r2'].mean():.3f}")
    ganador = min((n for n in comparacion if not es_linea_base(n)), key=lambda n: comparacion[n]["cv_rmse"])
    log(f"\nModelo elegido: {ganador}")
    modelo = candidatos()[ganador].fit(X_train, y_train)

    metricas = evaluar(modelo, X_test, y_test, figuras)
    metricas["modelo"] = ganador
    metricas["cv_r2_media"] = comparacion[ganador]["cv_r2"]
    metricas["comparacion_cv"] = comparacion
    metricas["baseline_media"] = metricas_regresion(y_test, candidatos()["precio medio (línea base)"].fit(X_train, y_train).predict(X_test))
    metricas["baseline_depreciacion"] = metricas_regresion(y_test, candidatos()["regla de depreciación (línea base)"].fit(X_train, y_train).predict(X_test))
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)
    # Solo informativo (la selección ya se hizo con la validación cruzada): cómo les va a los demás candidatos en la prueba de 60 autos.
    metricas["prueba_de_los_demas_candidatos"] = {n: metricas_regresion(y_test, candidatos()[n].fit(X_train, y_train).predict(X_test))
                                                  for n in candidatos() if n != ganador}

    # Experimento de extrapolación: se entrena solo con autos de precio de agencia <= PRECIO_ALTO y se prueba con los más caros.
    barato = df.present_price <= PRECIO_ALTO
    experimento = {"autos_de_entrenamiento": int(barato.sum()), "autos_caros_de_prueba": int((~barato).sum())}
    for nombre in ("random forest", ganador):
        ajustado = candidatos()[nombre].fit(df[barato][VARIABLES], df[barato][OBJETIVO])
        pred, real = ajustado.predict(df[~barato][VARIABLES]), df[~barato][OBJETIVO]
        experimento[nombre] = {"rmse": round(float(np.sqrt(mean_squared_error(real, pred))), 3), "sesgo": round(float((pred - real).mean()), 3)}
    metricas["experimento_extrapolacion"] = experimento
    log(f"Extrapolación a autos caros: {experimento}")

    # Despliegue: las métricas son las de la partición, pero con tan pocos datos el modelo que sirve la API se reentrena con todos.
    final = candidatos()[ganador].fit(df[VARIABLES], df[OBJETIVO])
    return {"pipeline": final, "metricas": metricas, "rango": {c: [float(df[c].min()), float(df[c].max())] for c in CON_RANGO},
            "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0]}


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    crudo = cargar_datos(quitar_duplicados=False)
    df = cargar_datos()
    entender(df, len(crudo))
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"], variables=VARIABLES, rango=r["rango"],
                   reentrenado_con_todo=True)
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k not in ("comparacion_cv", "ic95")})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

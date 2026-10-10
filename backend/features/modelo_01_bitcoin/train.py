"""Modelo 01 · Predicción del precio del Bitcoin.

Ejecutar desde backend/:  python -m features.modelo_01_bitcoin.train
Dataset: https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1 (precios diarios del 28-abr-2013 al 31-jul-2017)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, metricas_regresion
from features.modelo_01_bitcoin.serie import CARACTERISTICAS, VENTANA_MINIMA, caracteristicas, recursiva

CARPETA = Path(__file__).parent
OBJETIVO = "cierre"
SEMILLA = 42
FRACCION_PRUEBA = 0.20  # últimos días: se evalúa prediciendo el futuro, no días al azar
HORIZONTES = (1, 3, 7)  # días adelante que se evalúan; la API permite 1–7
MAX_HORIZONTE = max(HORIZONTES)
PLIEGUES_CV = 5
BLOQUE_BOOTSTRAP = 14  # días por bloque al remuestrear: los errores de días vecinos están correlacionados

# 1. Análisis del problema: estimar el precio de cierre del Bitcoin en los próximos días a partir de su propio historial.
#    Regresión sobre una serie de tiempo; se predice el retorno logarítmico del día siguiente y se encadena (predicción
#    recursiva) para ir más allá de un día. La referencia obligada es la **persistencia** (mañana = hoy). Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo (viene de la fecha más reciente a la más antigua) y lo deja ordenado y con números."""
    df = pd.read_csv(CARPETA / "dataset.csv")
    df["fecha"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    df["volumen"] = pd.to_numeric(df["Volume"].str.replace(",", "").replace("-", np.nan))
    df = df.rename(columns={"Close": OBJETIVO, "Open": "apertura", "High": "maximo", "Low": "minimo"})
    return df[["fecha", "apertura", "maximo", "minimo", OBJETIVO, "volumen"]].sort_values("fecha").reset_index(drop=True)


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    huecos = df.fecha.diff().dt.days.dropna()
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Rango: {df.fecha.min().date()} a {df.fecha.max().date()}")
    print(f"Días entre registros consecutivos: {huecos.value_counts().to_dict()} | fechas repetidas: {int(df.fecha.duplicated().sum())}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:\n", df.isna().sum())
    print("\nResumen del cierre (USD):\n", df[OBJETIVO].describe().round(2))
    r = np.log(df[OBJETIVO]).diff().dropna()
    print(f"\nRetorno diario: media {r.mean():.4f}, desviación {r.std():.4f}, mínimo {r.min():.3f}, máximo {r.max():.3f}")


def autocorrelacion(x: np.ndarray, rezagos: int = 20) -> np.ndarray:
    """Autocorrelación muestral de los rezagos 1..`rezagos`."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    return np.array([np.sum(x[k:] * x[:-k]) / np.sum(x ** 2) for k in range(1, rezagos + 1)])


def explorar(df: pd.DataFrame, figuras: Path, corte: pd.Timestamp) -> None:
    """3. Exploración de los datos."""
    plt.semilogy(df.fecha, df[OBJETIVO])
    plt.axvline(corte, color="red", linestyle="--", label="inicio de la prueba")
    plt.legend()
    plt.ylabel("Cierre (USD, escala logarítmica)")
    plt.title("Precio de cierre del Bitcoin")
    plt.xticks(rotation=30)
    guardar_figura(figuras, "serie_precio")

    r = np.log(df[OBJETIVO]).diff().dropna()
    plt.hist(r, bins=80)
    plt.xlabel("Retorno logarítmico diario")
    plt.title("Distribución de los retornos diarios (colas pesadas)")
    guardar_figura(figuras, "distribucion_retornos")

    rezagos = np.arange(1, 21)
    cota = 1.96 / np.sqrt(len(r))
    plt.bar(rezagos - 0.2, autocorrelacion(r.to_numpy()), width=0.4, label="retornos")
    plt.bar(rezagos + 0.2, autocorrelacion(r.to_numpy() ** 2), width=0.4, label="retornos al cuadrado")
    plt.axhline(cota, color="gray", linestyle="--")
    plt.axhline(-cota, color="gray", linestyle="--")
    plt.xlabel("Rezago (días)")
    plt.ylabel("Autocorrelación")
    plt.legend()
    plt.xticks(rezagos)
    plt.title("Autocorrelación: los retornos casi no, su tamaño sí")
    guardar_figura(figuras, "autocorrelacion")

    plt.plot(df.fecha, r.reindex(df.index).rolling(30).std())
    plt.ylabel("Desviación de los retornos (30 días)")
    plt.title("Volatilidad móvil del Bitcoin")
    plt.xticks(rotation=30)
    guardar_figura(figuras, "volatilidad")

    print("\nAutocorrelación de los retornos (rezagos 1-5):", autocorrelacion(r.to_numpy(), 5).round(3).tolist(), f"| cota 95 %: ±{cota:.3f}")
    print("Autocorrelación de los retornos al cuadrado (rezagos 1-5):", autocorrelacion(r.to_numpy() ** 2, 5).round(3).tolist())
    print(f"Días en que el precio sube: {(r > 0).mean():.1%} (todo el período)")
    print(f"Volumen: {int(df.volumen.isna().sum())} valores faltantes, todos antes de {df[df.volumen.notna()].fecha.min().date()}")


def construir(cierres: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    """Una fila por día de origen t: características con los cierres hasta t y como objetivo el retorno logarítmico t -> t+1.
    Solo quedan los orígenes con historial suficiente y con un día siguiente conocido."""
    X = caracteristicas(cierres)
    y = pd.Series(np.log(np.asarray(cierres, dtype=float))).diff().shift(-1)
    validas = X.notna().all(axis=1) & y.notna()
    return X[validas], y[validas]


def particion_temporal(n: int) -> tuple[int, int]:
    """Índice del primer día de la prueba y número de días. La prueba son los últimos FRACCION_PRUEBA de los días."""
    corte = int(n * (1 - FRACCION_PRUEBA))
    return corte, n - corte


def origenes(X: pd.DataFrame, corte: int) -> tuple[np.ndarray, np.ndarray]:
    """Orígenes de entrenamiento (su objetivo cae antes del corte) y de prueba (se parte del último día de entrenamiento).

    Ningún objetivo de entrenamiento mira dentro del período de prueba; las características de los orígenes de prueba
    solo usan datos ya ocurridos al iniciarlos.
    """
    indice = np.asarray(X.index)
    return indice[indice <= corte - 2], indice[indice >= corte - 1]


def armar(estimador, escalar: bool = False) -> Pipeline:
    """4. Modelo: (escalado opcional) + estimador sobre el retorno del día siguiente."""
    return Pipeline([("escala", StandardScaler() if escalar else "passthrough"), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Las dos primeras son líneas base (no se pueden elegir); las demás son modelos de aprendizaje automático."""
    return {
        "persistencia (línea base)": armar(DummyRegressor(strategy="constant", constant=0.0)),
        "deriva (línea base)": armar(DummyRegressor(strategy="mean")),
        "ridge": armar(RidgeCV(alphas=np.logspace(-1, 5, 25)), escalar=True),
        "random forest": armar(RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_depth=6,
                                                     random_state=SEMILLA, n_jobs=-1)),
        "gradient boosting": armar(HistGradientBoostingRegressor(max_iter=100, learning_rate=0.03, max_depth=3, min_samples_leaf=30,
                                                                 early_stopping=False, random_state=SEMILLA)),
    }


def es_linea_base(nombre: str) -> bool:
    return "línea base" in nombre


def intervalo_habilidad(e_modelo: np.ndarray, e_base: np.ndarray, remuestreos: int = 1000) -> list[float]:
    """IC 95 % de la habilidad 1 - RMSE(modelo)/RMSE(base) con bootstrap por bloques de días consecutivos."""
    n = len(e_modelo)
    rng = np.random.default_rng(SEMILLA)
    inicios = np.arange(n)
    habilidades = []
    for _ in range(remuestreos):
        elegido = np.concatenate([(rng.choice(inicios) + np.arange(BLOQUE_BOOTSTRAP)) % n for _ in range(int(np.ceil(n / BLOQUE_BOOTSTRAP)))])[:n]
        habilidades.append(1 - np.sqrt(np.mean(e_modelo[elegido] ** 2) / np.mean(e_base[elegido] ** 2)))
    return [round(float(np.percentile(habilidades, 2.5)), 4), round(float(np.percentile(habilidades, 97.5)), 4)]


def evaluar(pipeline: Pipeline, persistencia: Pipeline, deriva: Pipeline, cierres: np.ndarray, origenes_prueba: np.ndarray,
            sigma: float, figuras: Path | None) -> dict:
    """5. Evaluación en la prueba para 1, 3 y 7 días adelante: se parte de cada día de la prueba, se predice de forma
    recursiva y se compara con el cierre que realmente ocurrió."""
    origenes_prueba = origenes_prueba[origenes_prueba + MAX_HORIZONTE <= len(cierres) - 1]  # todos los horizontes con dato real
    historiales = [cierres[:t + 1] for t in origenes_prueba]
    pred = {nombre: recursiva(p, historiales, MAX_HORIZONTE) for nombre, p in
            (("modelo", pipeline), ("persistencia", persistencia), ("deriva", deriva))}
    real = np.array([[cierres[t + h] for h in range(1, MAX_HORIZONTE + 1)] for t in origenes_prueba])
    partida = np.array([cierres[t] for t in origenes_prueba])

    metricas: dict = {"horizontes": {}}
    for h in HORIZONTES:
        y, columna = real[:, h - 1], h - 1
        error = {nombre: y - p[:, columna] for nombre, p in pred.items()}
        entrada = {nombre: metricas_regresion(y, p[:, columna]) for nombre, p in pred.items()}
        for nombre, p in pred.items():
            entrada[nombre]["mape_pct"] = round(float(np.mean(np.abs(y - p[:, columna]) / y) * 100), 3)
        sube = y > partida
        entrada["modelo"]["acierto_direccion"] = round(float(np.mean((pred["modelo"][:, columna] > partida) == sube)), 4)
        entrada["siempre_sube"] = {"acierto_direccion": round(float(np.mean(sube)), 4)}
        lo = pred["modelo"][:, columna] * np.exp(-1.96 * sigma * np.sqrt(h))
        hi = pred["modelo"][:, columna] * np.exp(1.96 * sigma * np.sqrt(h))
        entrada["cobertura_intervalo_95"] = round(float(np.mean((y >= lo) & (y <= hi))), 4)
        entrada["habilidad_frente_a_persistencia"] = round(float(1 - np.sqrt(np.mean(error["modelo"] ** 2) / np.mean(error["persistencia"] ** 2))), 4)
        entrada["habilidad_ic95"] = intervalo_habilidad(error["modelo"], error["persistencia"])
        entrada["n_origenes"] = len(y)
        metricas["horizontes"][str(h)] = entrada

    if figuras is not None:
        fechas = np.arange(len(origenes_prueba))
        for h in (1, 7):
            plt.plot(fechas + h, real[:, h - 1], label="real", color="black")
            plt.plot(fechas + h, pred["modelo"][:, h - 1], label="modelo", linestyle="--")
            plt.plot(fechas + h, pred["persistencia"][:, h - 1], label="persistencia", linestyle=":")
            plt.xlabel("Días desde el inicio de la prueba")
            plt.ylabel("Cierre (USD)")
            plt.legend()
            plt.title(f"Prueba: cierre real y predicho {h} día(s) adelante")
            guardar_figura(figuras, f"prueba_{h}_dia{'s' if h > 1 else ''}")
        plt.bar([str(h) for h in HORIZONTES], [metricas["horizontes"][str(h)]["habilidad_frente_a_persistencia"] * 100 for h in HORIZONTES])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Días adelante")
        plt.ylabel("Habilidad frente a la persistencia (%)")
        plt.title("Mejora del RMSE respecto de repetir el último precio")
        guardar_figura(figuras, "habilidad_por_horizonte")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    cierres = df[OBJETIVO].to_numpy(dtype=float)
    X, y = construir(cierres)
    corte, _ = particion_temporal(len(cierres))
    entrena, prueba = origenes(X, corte)
    X_train, y_train = X.loc[entrena], y.loc[entrena]
    log(f"Entrenamiento: {len(entrena)} orígenes hasta {df.fecha[corte - 1].date()} | prueba: desde {df.fecha[corte].date()}")

    # La selección usa solo el entrenamiento: validación cruzada de ventana creciente (TimeSeriesSplit) sobre el RMSE
    # del retorno del día siguiente. El precio de un modelo recursivo se evalúa después, en la prueba.
    comparacion = {}
    for nombre, pipe in candidatos().items():
        rmse = -cross_val_score(pipe, X_train, y_train, cv=TimeSeriesSplit(PLIEGUES_CV), scoring="neg_root_mean_squared_error", error_score="raise")
        comparacion[nombre] = {"cv_rmse_retorno": round(float(rmse.mean()), 5), "cv_rmse_desv": round(float(rmse.std()), 5)}
        log(f"{nombre:28s} RMSE del retorno diario (cv) = {rmse.mean():.5f} ± {rmse.std():.5f}")
    ganador = min((n for n in comparacion if not es_linea_base(n)), key=lambda n: comparacion[n]["cv_rmse_retorno"])
    mejor_base = min((n for n in comparacion if es_linea_base(n)), key=lambda n: comparacion[n]["cv_rmse_retorno"])
    log(f"\nModelo elegido entre los de aprendizaje automático: {ganador} | mejor línea base en cv: {mejor_base}")

    pipeline = candidatos()[ganador].fit(X_train, y_train)
    persistencia = candidatos()["persistencia (línea base)"].fit(X_train, y_train)
    deriva = candidatos()["deriva (línea base)"].fit(X_train, y_train)
    sigma = float(np.std(y_train - pipeline.predict(X_train), ddof=1))  # desviación del retorno diario: da el intervalo del 95 %

    metricas = evaluar(pipeline, persistencia, deriva, cierres, prueba, sigma, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    metricas["mejor_linea_base_cv"] = mejor_base
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(entrena), len(cierres) - corte
    metricas["sigma_retorno_diario"] = round(sigma, 5)
    metricas["fecha_inicio_prueba"] = str(df.fecha[corte].date())
    metricas["prueba_sube_x"] = round(float(cierres[-1] / cierres[corte]), 2)
    metricas["retorno_medio_dia_entrenamiento"] = round(float(y_train.mean()), 5)
    log("\nHorizontes de la prueba:")
    for h, m in metricas["horizontes"].items():
        log(f"  {h} día(s): modelo RMSE {m['modelo']['rmse']} | persistencia {m['persistencia']['rmse']} | habilidad {m['habilidad_frente_a_persistencia']:+.2%} IC95 {m['habilidad_ic95']}")

    # Despliegue: las métricas son las de la partición temporal, pero el modelo que sirve la API se reentrena con TODOS los
    # días para que parta del cierre más reciente del dataset.
    final = candidatos()[ganador].fit(X, y)
    sigma_final = float(np.std(y - final.predict(X), ddof=1))
    return {"pipeline": final, "metricas": metricas, "sigma": round(sigma_final, 5),
            "ultimos_cierres": [float(v) for v in cierres[-VENTANA_MINIMA:]], "fecha_max": str(df.fecha.iloc[-1].date()),
            "entrada_ejemplo": {"dias_adelante": 1}}


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    corte, _ = particion_temporal(len(df))
    explorar(df, figuras, df.fecha[corte])
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], r["entrada_ejemplo"], variables=CARACTERISTICAS, sigma=r["sigma"],
                   ultimos_cierres=r["ultimos_cierres"], fecha_max=r["fecha_max"], reentrenado_con_todo=True)
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

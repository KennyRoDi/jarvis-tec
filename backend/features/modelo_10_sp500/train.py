"""Modelo 10 · Predicción del precio de acciones del S&P 500.

Ejecutar desde backend/:  python -m features.modelo_10_sp500.train
Dataset: https://www.kaggle.com/camnugent/sandp500 (`all_stocks_5yr.csv`; aquí el recorte de AAPL, MSFT, AMZN y GOOGL)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
Mismo enfoque que el modelo 01 (retorno logarítmico del día siguiente, predicción recursiva, persistencia como referencia), con
una serie por símbolo y **un solo modelo compartido** por los cuatro.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo
from features.modelo_10_sp500.serie import CARACTERISTICAS, VENTANA_MINIMA, caracteristicas, recursiva

CARPETA = Path(__file__).parent
OBJETIVO = "cierre"
SIMBOLOS = ["AAPL", "MSFT", "AMZN", "GOOGL"]
SEMILLA = 42
FRACCION_PRUEBA = 0.20  # últimos días: se evalúa prediciendo el futuro, no días al azar
HORIZONTES = (1, 3, 7)  # días hábiles adelante que se evalúan; la API permite 1–7
MAX_HORIZONTE = max(HORIZONTES)
PLIEGUES_CV = 5
BLOQUE_BOOTSTRAP = 14  # días por bloque al remuestrear: los errores de días vecinos están correlacionados

# 1. Análisis del problema: estimar el precio de cierre de una acción del S&P 500 (AAPL, MSFT, AMZN o GOOGL) en los próximos días
#    hábiles. Regresión sobre series de tiempo (una por símbolo) con un modelo compartido que predice el retorno logarítmico del día
#    siguiente y se encadena (predicción recursiva). La referencia obligada es la persistencia (mañana = hoy). Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo y lo deja ordenado por símbolo y fecha."""
    df = pd.read_csv(CARPETA / "dataset.csv", parse_dates=["date"])
    df = df.rename(columns={"date": "fecha", "Name": "simbolo", "close": OBJETIVO, "open": "apertura", "high": "maximo", "low": "minimo",
                            "volume": "volumen"})
    return df.sort_values(["simbolo", "fecha"]).reset_index(drop=True)


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Símbolos: {df.simbolo.nunique()}  Fechas: {df.fecha.nunique()}")
    print(f"Rango: {df.fecha.min().date()} a {df.fecha.max().date()} | filas por símbolo: {df.simbolo.value_counts().sort_index().to_dict()}")
    huecos = df[df.simbolo == SIMBOLOS[0]].fecha.diff().dt.days.dropna()
    print(f"Días naturales entre sesiones consecutivas: {huecos.value_counts().sort_index().to_dict()} | duplicados (fecha, símbolo): {int(df.duplicated(['fecha', 'simbolo']).sum())}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:\n", df.isna().sum())
    print("\nCierre por símbolo (USD):\n", df.groupby("simbolo")[OBJETIVO].describe().round(2))
    r = retornos(df)
    print("\nRetorno diario por símbolo (media, desviación):\n", r.groupby(df.simbolo).agg(["mean", "std"]).round(4))


def retornos(df: pd.DataFrame) -> pd.Series:
    """Retorno logarítmico diario de cada símbolo (el primer día de cada uno queda vacío)."""
    return np.log(df[OBJETIVO]).groupby(df.simbolo).diff()


def autocorrelacion(x: np.ndarray, rezagos: int = 20) -> np.ndarray:
    """Autocorrelación muestral de los rezagos 1..`rezagos`."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    return np.array([np.sum(x[k:] * x[:-k]) / np.sum(x ** 2) for k in range(1, rezagos + 1)])


def explorar(df: pd.DataFrame, figuras: Path, corte: pd.Timestamp) -> None:
    """3. Exploración de los datos."""
    base = df.pivot(index="fecha", columns="simbolo", values=OBJETIVO)
    (100 * base / base.iloc[0]).plot()
    plt.axvline(corte, color="red", linestyle="--", label="inicio de la prueba")
    plt.legend()
    plt.ylabel("Cierre (base 100 = 8-feb-2013)")
    plt.title("Evolución de los cuatro símbolos")
    guardar_figura(figuras, "series_normalizadas")

    r = retornos(df).dropna()
    plt.boxplot([r[df.loc[r.index, "simbolo"] == simbolo] for simbolo in SIMBOLOS], tick_labels=SIMBOLOS)
    plt.xlabel("Símbolo")
    plt.ylabel("Retorno logarítmico diario")
    plt.title("Retornos diarios: colas pesadas de distinto grosor y volatilidad")
    guardar_figura(figuras, "retornos_por_simbolo")

    ret = retornos(df).to_frame("r").assign(fecha=df.fecha, simbolo=df.simbolo).pivot(index="fecha", columns="simbolo", values="r").dropna()
    sns.heatmap(ret.corr(), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Correlación entre los retornos diarios de los símbolos")
    guardar_figura(figuras, "correlacion_retornos")

    rezagos = np.arange(1, 21)
    medio = np.mean([autocorrelacion(ret[s].to_numpy()) for s in ret.columns], axis=0)
    medio_cuadrado = np.mean([autocorrelacion(ret[s].to_numpy() ** 2) for s in ret.columns], axis=0)
    plt.bar(rezagos - 0.2, medio, width=0.4, label="retornos")
    plt.bar(rezagos + 0.2, medio_cuadrado, width=0.4, label="retornos al cuadrado")
    cota = 1.96 / np.sqrt(len(ret))
    plt.axhline(cota, color="gray", linestyle="--")
    plt.axhline(-cota, color="gray", linestyle="--")
    plt.xticks(rezagos)
    plt.xlabel("Rezago (sesiones)")
    plt.ylabel("Autocorrelación (promedio de los símbolos)")
    plt.legend()
    plt.title("Autocorrelación: los retornos casi no, su tamaño sí")
    guardar_figura(figuras, "autocorrelacion")

    print("\nAutocorrelación media de los retornos (rezagos 1-5):", medio[:5].round(3).tolist(), f"| cota 95 % por símbolo: ±{cota:.3f}")
    print("Autocorrelación media de los retornos al cuadrado (rezagos 1-5):", medio_cuadrado[:5].round(3).tolist())
    print("Correlación media entre pares de símbolos:", round(float(ret.corr().to_numpy()[np.triu_indices(len(ret.columns), 1)].mean()), 3))
    print(f"Sesiones en que el precio sube: {(r > 0).mean():.1%} (todos los símbolos)")


def construir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Una fila por (símbolo, día de origen t): características con los cierres hasta t y como objetivo el retorno logarítmico t -> t+1.
    El índice es (símbolo, posición dentro de la serie). Solo quedan los orígenes con historial suficiente y día siguiente conocido."""
    bloques_x, bloques_y = [], []
    for simbolo, g in df.groupby("simbolo", sort=True):
        cierres = g[OBJETIVO].to_numpy(dtype=float)
        X = caracteristicas(cierres)
        y = pd.Series(np.log(cierres)).diff().shift(-1)
        validas = X.notna().all(axis=1) & y.notna()
        X, y = X[validas], y[validas]
        X.index = y.index = pd.MultiIndex.from_product([[simbolo], X.index], names=["simbolo", "posicion"])
        bloques_x.append(X)
        bloques_y.append(y)
    return pd.concat(bloques_x), pd.concat(bloques_y)


def particion_temporal(df: pd.DataFrame) -> int:
    """Posición (dentro de cada serie) del primer día de la prueba. Los cuatro símbolos comparten las mismas fechas;
    la prueba son los últimos FRACCION_PRUEBA de las sesiones."""
    fechas = df.groupby("simbolo").fecha.apply(list)
    assert all(f == fechas.iloc[0] for f in fechas), "los símbolos deben tener las mismas fechas"
    return int(len(fechas.iloc[0]) * (1 - FRACCION_PRUEBA))


def origenes(X: pd.DataFrame, corte: int) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de entrenamiento en `X` (su objetivo cae antes del corte) y de prueba (se parte del último día de entrenamiento)."""
    posicion = X.index.get_level_values("posicion").to_numpy()
    return np.flatnonzero(posicion <= corte - 2), np.flatnonzero(posicion >= corte - 1)


def pliegues_por_fecha(X: pd.DataFrame, indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación cruzada de ventana creciente por **fechas**: las filas de un mismo día (los cuatro símbolos, muy correlacionados) van juntas."""
    posicion = X.index.get_level_values("posicion").to_numpy()[indices]
    dias = np.sort(np.unique(posicion))
    pliegues = []
    for entrena, prueba in TimeSeriesSplit(n_splits=PLIEGUES_CV).split(dias):
        pliegues.append((np.flatnonzero(np.isin(posicion, dias[entrena])), np.flatnonzero(np.isin(posicion, dias[prueba]))))
    return pliegues


def armar(estimador, escalar: bool = False) -> Pipeline:
    """4. Modelo: (escalado opcional) + estimador sobre el retorno del día siguiente."""
    return Pipeline([("escala", StandardScaler() if escalar else "passthrough"), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Las dos primeras son líneas base (no se pueden elegir); las demás son modelos de aprendizaje automático."""
    return {
        "persistencia (línea base)": armar(DummyRegressor(strategy="constant", constant=0.0)),
        "deriva (línea base)": armar(DummyRegressor(strategy="mean")),
        "ridge": armar(RidgeCV(alphas=np.logspace(-1, 5, 25)), escalar=True),
        "random forest": armar(RandomForestRegressor(n_estimators=200, min_samples_leaf=50, max_depth=6,
                                                     random_state=SEMILLA, n_jobs=-1)),
        "gradient boosting": armar(HistGradientBoostingRegressor(max_iter=100, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
                                                                 early_stopping=False, random_state=SEMILLA)),
    }


def es_linea_base(nombre: str) -> bool:
    return "línea base" in nombre


def intervalo_habilidad(e_modelo: np.ndarray, e_base: np.ndarray, remuestreos: int = 1000) -> list[float]:
    """IC 95 % de la habilidad 1 - RMSE(modelo)/RMSE(base) con bootstrap por bloques circulares de días consecutivos.
    Los errores son matrices (días × símbolos): se remuestrean días completos para conservar la correlación entre símbolos."""
    n = len(e_modelo)
    rng = np.random.default_rng(SEMILLA)
    habilidades = []
    for _ in range(remuestreos):
        elegido = np.concatenate([(rng.integers(n) + np.arange(BLOQUE_BOOTSTRAP)) % n for _ in range(int(np.ceil(n / BLOQUE_BOOTSTRAP)))])[:n]
        habilidades.append(1 - np.sqrt(np.mean(e_modelo[elegido] ** 2) / np.mean(e_base[elegido] ** 2)))
    return [round(float(np.percentile(habilidades, 2.5)), 4), round(float(np.percentile(habilidades, 97.5)), 4)]


def habilidad(e_modelo, e_base) -> float:
    return round(float(1 - np.sqrt(np.mean(np.square(e_modelo)) / np.mean(np.square(e_base)))), 4)


def evaluar(pipeline: Pipeline, persistencia: Pipeline, deriva: Pipeline, df: pd.DataFrame, corte: int, sigma: dict, figuras: Path | None) -> dict:
    """5. Evaluación en la prueba para 1, 3 y 7 sesiones adelante: se parte de cada día de la prueba, se predice de forma recursiva y
    se compara con el cierre que realmente ocurrió. Los errores se miden en proporción del precio real para poder juntar símbolos."""
    cierres = {s: g[OBJETIVO].to_numpy(dtype=float) for s, g in df.groupby("simbolo", sort=True)}
    n = len(next(iter(cierres.values())))
    origenes_prueba = np.arange(corte - 1, n - MAX_HORIZONTE)  # todos los horizontes con dato real
    pred = {nombre: {s: recursiva(p, [c[:t + 1] for t in origenes_prueba], MAX_HORIZONTE) for s, c in cierres.items()}
            for nombre, p in (("modelo", pipeline), ("persistencia", persistencia), ("deriva", deriva))}
    real = {s: np.array([[c[t + h] for h in range(1, MAX_HORIZONTE + 1)] for t in origenes_prueba]) for s, c in cierres.items()}
    partida = {s: c[origenes_prueba] for s, c in cierres.items()}
    simbolos = sorted(cierres)

    metricas: dict = {"horizontes": {}}
    for h in HORIZONTES:
        columna = h - 1
        y = {s: real[s][:, columna] for s in simbolos}
        # errores relativos (días × símbolos)
        rel = {nombre: np.column_stack([(y[s] - pred[nombre][s][:, columna]) / y[s] for s in simbolos]) for nombre in pred}
        sube = {s: y[s] > partida[s] for s in simbolos}
        entrada: dict = {"modelo": {}, "persistencia": {}, "deriva": {}}
        for nombre in pred:
            entrada[nombre]["rmse_relativo_pct"] = round(float(np.sqrt(np.mean(rel[nombre] ** 2)) * 100), 4)
            entrada[nombre]["mape_pct"] = round(float(np.mean(np.abs(rel[nombre])) * 100), 4)
        entrada["modelo"]["acierto_direccion"] = round(float(np.mean([np.mean((pred["modelo"][s][:, columna] > partida[s]) == sube[s]) for s in simbolos])), 4)
        entrada["siempre_sube"] = {"acierto_direccion": round(float(np.mean([np.mean(sube[s]) for s in simbolos])), 4)}
        dentro = {}
        for s in simbolos:
            amplitud = np.exp(1.96 * sigma[s] * np.sqrt(h))
            dentro[s] = (y[s] >= pred["modelo"][s][:, columna] / amplitud) & (y[s] <= pred["modelo"][s][:, columna] * amplitud)
        cobertura = {s: round(float(np.mean(d)), 4) for s, d in dentro.items()}
        entrada["cobertura_intervalo_95"] = {"global": round(float(np.mean(list(cobertura.values()))), 4), **cobertura}
        mitad = len(origenes_prueba) // 2
        entrada["cobertura_intervalo_95_por_mitad"] = [round(float(np.mean([d[:mitad].mean() for d in dentro.values()])), 4),
                                                       round(float(np.mean([d[mitad:].mean() for d in dentro.values()])), 4)]
        entrada["habilidad_frente_a_persistencia"] = habilidad(rel["modelo"], rel["persistencia"])
        entrada["habilidad_ic95"] = intervalo_habilidad(rel["modelo"], rel["persistencia"])
        entrada["habilidad_frente_a_deriva"] = habilidad(rel["modelo"], rel["deriva"])
        entrada["habilidad_frente_a_deriva_ic95"] = intervalo_habilidad(rel["modelo"], rel["deriva"])
        entrada["habilidad_deriva_frente_a_persistencia"] = habilidad(rel["deriva"], rel["persistencia"])
        entrada["habilidad_por_simbolo"] = {s: habilidad(rel["modelo"][:, i], rel["persistencia"][:, i]) for i, s in enumerate(simbolos)}
        entrada["por_simbolo_usd"] = {s: {"rmse_modelo": round(float(np.sqrt(np.mean((y[s] - pred["modelo"][s][:, columna]) ** 2))), 3),
                                          "rmse_persistencia": round(float(np.sqrt(np.mean((y[s] - pred["persistencia"][s][:, columna]) ** 2))), 3)}
                                      for s in simbolos}
        entrada["n_origenes"] = len(origenes_prueba)  # por símbolo
        metricas["horizontes"][str(h)] = entrada

    if figuras is not None:
        fig, ejes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for eje, s in zip(ejes.ravel(), simbolos):
            eje.plot(origenes_prueba + 1, real[s][:, 0], color="black", label="real")
            eje.plot(origenes_prueba + 1, pred["modelo"][s][:, 0], linestyle="--", label="modelo (1 día)")
            eje.plot(origenes_prueba + 1, pred["persistencia"][s][:, 0], linestyle=":", label="persistencia")
            eje.set_title(s)
            eje.set_ylabel("Cierre (USD)")
        ejes[0, 0].legend()
        fig.suptitle("Prueba: cierre real y predicho un día adelante")
        guardar_figura(figuras, "prueba_1_dia")
        plt.bar([str(h) for h in HORIZONTES], [metricas["horizontes"][str(h)]["habilidad_frente_a_persistencia"] * 100 for h in HORIZONTES])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Sesiones adelante")
        plt.ylabel("Habilidad frente a la persistencia (%)")
        plt.title("Mejora del error relativo respecto de repetir el último precio")
        guardar_figura(figuras, "habilidad_por_horizonte")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    X, y = construir(df)
    corte = particion_temporal(df)
    entrena, _ = origenes(X, corte)
    X_train, y_train = X.iloc[entrena], y.iloc[entrena]
    fechas = df[df.simbolo == SIMBOLOS[0]].fecha.reset_index(drop=True)
    log(f"Entrenamiento: {len(entrena)} filas (símbolo, día) hasta {fechas[corte - 1].date()} | prueba: desde {fechas[corte].date()}")

    # La selección usa solo el entrenamiento: validación cruzada de ventana creciente por fechas sobre el RMSE del retorno diario.
    pliegues = pliegues_por_fecha(X_train, np.arange(len(X_train)))
    comparacion = {}
    for nombre, pipe in candidatos().items():
        rmse = -cross_val_score(pipe, X_train, y_train, cv=pliegues, scoring="neg_root_mean_squared_error", error_score="raise")
        comparacion[nombre] = {"cv_rmse_retorno": round(float(rmse.mean()), 5), "cv_rmse_desv": round(float(rmse.std()), 5)}
        log(f"{nombre:28s} RMSE del retorno diario (cv) = {rmse.mean():.5f} ± {rmse.std():.5f}")
    ganador = min((n for n in comparacion if not es_linea_base(n)), key=lambda n: comparacion[n]["cv_rmse_retorno"])
    mejor_base = min((n for n in comparacion if es_linea_base(n)), key=lambda n: comparacion[n]["cv_rmse_retorno"])
    log(f"\nModelo elegido entre los de aprendizaje automático: {ganador} | mejor línea base en cv: {mejor_base}")

    pipeline = candidatos()[ganador].fit(X_train, y_train)
    persistencia = candidatos()["persistencia (línea base)"].fit(X_train, y_train)
    deriva = candidatos()["deriva (línea base)"].fit(X_train, y_train)
    residuo = y_train - pipeline.predict(X_train)
    sigma = {s: float(residuo.xs(s, level="simbolo").std(ddof=1)) for s in SIMBOLOS}  # volatilidad diaria de cada símbolo: da su intervalo del 95 %

    metricas = evaluar(pipeline, persistencia, deriva, df, corte, sigma, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    metricas["mejor_linea_base_cv"] = mejor_base
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(entrena), len(fechas) - corte
    metricas["sigma_retorno_diario"] = {s: round(v, 5) for s, v in sigma.items()}
    metricas["fecha_inicio_prueba"] = str(fechas[corte].date())
    metricas["retorno_medio_dia_entrenamiento"] = round(float(y_train.mean()), 5)
    log("\nHorizontes de la prueba (error relativo, todos los símbolos):")
    for h, m in metricas["horizontes"].items():
        log(f"  {h} día(s): modelo {m['modelo']['rmse_relativo_pct']} % | persistencia {m['persistencia']['rmse_relativo_pct']} % | "
            f"habilidad {m['habilidad_frente_a_persistencia']:+.2%} IC95 {m['habilidad_ic95']} | frente a la deriva {m['habilidad_frente_a_deriva']:+.2%} IC95 {m['habilidad_frente_a_deriva_ic95']} | por símbolo {m['habilidad_por_simbolo']}")

    # Despliegue: las métricas son las de la partición temporal, pero el modelo que sirve la API se reentrena con TODOS los
    # días para que parta del cierre más reciente de cada símbolo.
    final = candidatos()[ganador].fit(X, y)
    residuo_final = y - final.predict(X)
    return {"pipeline": final, "metricas": metricas,
            "sigma": {s: round(float(residuo_final.xs(s, level="simbolo").std(ddof=1)), 5) for s in SIMBOLOS},
            "ultimos_cierres": {s: [float(v) for v in g[OBJETIVO].to_numpy()[-VENTANA_MINIMA:]] for s, g in df.groupby("simbolo", sort=True)},
            "fecha_max": str(df.fecha.max().date()), "entrada_ejemplo": {"simbolo": "AAPL", "dias_adelante": 1}}


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    corte = particion_temporal(df)
    explorar(df, figuras, df[df.simbolo == SIMBOLOS[0]].fecha.reset_index(drop=True)[corte])
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], r["entrada_ejemplo"], variables=CARACTERISTICAS, sigma=r["sigma"],
                   ultimos_cierres=r["ultimos_cierres"], fecha_max=r["fecha_max"], simbolos=SIMBOLOS, reentrenado_con_todo=True)
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()

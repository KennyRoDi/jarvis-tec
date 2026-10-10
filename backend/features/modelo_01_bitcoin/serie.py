"""Características de la serie de precios del Bitcoin y predicción recursiva a varios días.

Vive en su propio módulo (y no en train.py) porque lo usan el entrenamiento, la evaluación y el router, y para que joblib
pueda cargar el modelo. Todas las características se calculan **solo con cierres pasados**: así se pueden encadenar
predicciones (el cierre predicho de mañana entra al cálculo de pasado mañana).
"""
import numpy as np
import pandas as pd

VENTANA_MINIMA = 31  # cierres necesarios para calcular todas las características
CARACTERISTICAS = ["ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_7", "ret_30", "dist_media_7", "dist_media_30", "vol_7", "vol_30"]


def caracteristicas(cierres) -> pd.DataFrame:
    """Una fila por día usando solo los cierres hasta ese día. Las primeras VENTANA_MINIMA - 1 filas son NaN."""
    c = pd.Series(np.asarray(cierres, dtype=float))
    lc = np.log(c)
    r = lc.diff()
    return pd.DataFrame({
        "ret_1": r, "ret_2": r.shift(1), "ret_3": r.shift(2), "ret_4": r.shift(3), "ret_5": r.shift(4),
        "ret_7": lc - lc.shift(7), "ret_30": lc - lc.shift(30),
        "dist_media_7": lc - np.log(c.rolling(7).mean()), "dist_media_30": lc - np.log(c.rolling(30).mean()),
        "vol_7": r.rolling(7).std(), "vol_30": r.rolling(30).std(),
    })[CARACTERISTICAS]


def ultima_fila(cierres) -> dict:
    """Las mismas características, solo para el último día (rápido: lo usa la predicción recursiva y la API)."""
    c = np.asarray(cierres, dtype=float)[-VENTANA_MINIMA:]
    if len(c) < VENTANA_MINIMA:
        raise ValueError(f"se necesitan al menos {VENTANA_MINIMA} cierres")
    lc = np.log(c)
    r = np.diff(lc)
    return {
        "ret_1": r[-1], "ret_2": r[-2], "ret_3": r[-3], "ret_4": r[-4], "ret_5": r[-5],
        "ret_7": lc[-1] - lc[-8], "ret_30": lc[-1] - lc[-31],
        "dist_media_7": lc[-1] - np.log(c[-7:].mean()), "dist_media_30": lc[-1] - np.log(c[-30:].mean()),
        "vol_7": r[-7:].std(ddof=1), "vol_30": r[-30:].std(ddof=1),
    }


def recursiva(pipeline, historiales: list, pasos: int) -> np.ndarray:
    """Predice el cierre de los próximos `pasos` días para cada historial: matriz (n_historiales, pasos).

    Un paso predice el retorno logarítmico del día siguiente; el cierre resultante se agrega al historial y se repite.
    """
    historiales = [np.asarray(h, dtype=float)[-VENTANA_MINIMA:].copy() for h in historiales]
    salida = np.empty((len(historiales), pasos))
    for paso in range(pasos):
        X = pd.DataFrame([ultima_fila(h) for h in historiales], columns=CARACTERISTICAS)
        retornos = pipeline.predict(X)
        for i, h in enumerate(historiales):
            salida[i, paso] = h[-1] * np.exp(retornos[i])
            historiales[i] = np.append(h, salida[i, paso])[-VENTANA_MINIMA:]
    return salida

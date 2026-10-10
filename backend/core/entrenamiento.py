"""Utilidades compartidas por los train.py: métricas, figuras y guardado de artefactos."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # sin ventana: las figuras se guardan como PNG para Overleaf
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics  # noqa: E402

from core.modelos import ARTEFACTO, METRICAS  # noqa: E402


def carpeta_figuras(carpeta: Path) -> Path:
    destino = carpeta / "figuras"
    destino.mkdir(exist_ok=True)
    return destino


def guardar_figura(destino: Path, nombre: str) -> None:
    plt.tight_layout()
    plt.savefig(destino / f"{nombre}.png", dpi=150)
    plt.close()


def metricas_regresion(y_real, y_pred) -> dict:
    return {
        "r2": round(float(metrics.r2_score(y_real, y_pred)), 4),
        "mae": round(float(metrics.mean_absolute_error(y_real, y_pred)), 4),
        "rmse": round(float(np.sqrt(metrics.mean_squared_error(y_real, y_pred))), 4),
    }


def metricas_clasificacion(y_real, y_pred, orden=None) -> dict:
    """`orden`: clases en su orden natural para la matriz de confusión (filas = real, columnas = predicho).
    Sin él, scikit-learn las ordena alfabéticamente."""
    return {
        "accuracy": round(float(metrics.accuracy_score(y_real, y_pred)), 4),
        "precision_macro": round(float(metrics.precision_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(metrics.recall_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(metrics.f1_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "matriz_confusion": metrics.confusion_matrix(y_real, y_pred, labels=orden).tolist(),
        **({"orden_clases": list(orden)} if orden is not None else {}),
    }


def umbral_optimo_f1(y_real, probabilidad, rejilla=None) -> float:
    """Umbral de probabilidad que maximiza el F1 de la clase positiva (`y_real` con 1 = positivo).

    Debe calcularse con probabilidades **fuera de muestra** (p. ej. `cross_val_predict` sobre el entrenamiento),
    nunca con el conjunto de prueba. En caso de empate gana el umbral más bajo.
    """
    y = np.asarray(y_real).astype(bool)
    rejilla = np.round(np.arange(0.05, 0.951, 0.01), 2) if rejilla is None else rejilla  # redondeada: sin ruido de coma flotante
    puntajes = [metrics.f1_score(y, np.asarray(probabilidad) >= t, zero_division=0) for t in rejilla]
    return round(float(rejilla[int(np.argmax(puntajes))]), 2)


def umbral_para_recall(y_real, probabilidad, objetivo: float = 0.8, rejilla=None) -> float:
    """Mayor umbral con el que el recall de la clase positiva (`y_real` con 1 = positivo) alcanza `objetivo`.

    Sirve para un tamizaje en el que importa no dejar casos sin detectar. Como `umbral_optimo_f1`, debe calcularse
    con probabilidades fuera de muestra. Si ningún umbral alcanza el objetivo devuelve el más bajo de la rejilla.
    """
    y = np.asarray(y_real).astype(bool)
    p = np.asarray(probabilidad)
    rejilla = np.round(np.arange(0.005, 0.951, 0.005), 3) if rejilla is None else rejilla
    for umbral in sorted(rejilla, reverse=True):
        if (p[y] >= umbral).mean() >= objetivo:
            return round(float(umbral), 3)
    return round(float(min(rejilla)), 3)


def intervalo_bootstrap(y_real, probabilidad, funcion, remuestreos: int = 1000, semilla: int = 42) -> list[float]:
    """Intervalo de confianza del 95 % (percentiles 2.5 y 97.5) de `funcion(y, p)` por remuestreo con reemplazo.

    Con pocos positivos en la prueba las métricas son muy ruidosas: esto cuantifica cuánto.
    """
    y, p = np.asarray(y_real), np.asarray(probabilidad)
    rng = np.random.default_rng(semilla)
    valores = []
    while len(valores) < remuestreos:
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:  # hace falta al menos un positivo y un negativo
            valores.append(funcion(y[i], p[i]))
    return [round(float(np.percentile(valores, 2.5)), 4), round(float(np.percentile(valores, 97.5)), 4)]


def guardar_modelo(carpeta: Path, pipeline, metricas: dict, entrada_ejemplo: dict, **extra) -> None:
    """Guarda modelo.joblib (lo usa el router) y metricas.json (lo usa /info y analisis.md)."""
    entrada_ejemplo = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in entrada_ejemplo.items()}
    joblib.dump({"pipeline": pipeline, "metricas": metricas, **extra}, carpeta / ARTEFACTO, compress=3)
    contenido = {"metricas": metricas, "entrada_ejemplo": entrada_ejemplo}
    (carpeta / METRICAS).write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")

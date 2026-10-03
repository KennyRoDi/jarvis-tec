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


def metricas_clasificacion(y_real, y_pred) -> dict:
    return {
        "accuracy": round(float(metrics.accuracy_score(y_real, y_pred)), 4),
        "precision_macro": round(float(metrics.precision_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(metrics.recall_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(metrics.f1_score(y_real, y_pred, average="macro", zero_division=0)), 4),
        "matriz_confusion": metrics.confusion_matrix(y_real, y_pred).tolist(),
    }


def guardar_modelo(carpeta: Path, pipeline, metricas: dict, entrada_ejemplo: dict, **extra) -> None:
    """Guarda modelo.joblib (lo usa el router) y metricas.json (lo usa /info y analisis.md)."""
    entrada_ejemplo = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in entrada_ejemplo.items()}
    joblib.dump({"pipeline": pipeline, "metricas": metricas, **extra}, carpeta / ARTEFACTO, compress=3)
    contenido = {"metricas": metricas, "entrada_ejemplo": entrada_ejemplo}
    (carpeta / METRICAS).write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")

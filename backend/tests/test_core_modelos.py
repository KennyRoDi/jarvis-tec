"""Pruebas de las utilidades comunes de entrenamiento y predicción (backend/core)."""
import joblib
import numpy as np
import pytest

from core.entrenamiento import metricas_clasificacion
from core.modelos import fuera_de_rango


def test_matriz_de_confusion_respeta_el_orden_pedido():
    real = ["baja"] * 4 + ["media"] * 3 + ["alta"] * 2
    pred = ["baja"] * 3 + ["media"] + ["media"] * 3 + ["alta", "media"]
    metricas = metricas_clasificacion(real, pred, orden=["baja", "media", "alta"])
    assert metricas["matriz_confusion"] == [[3, 1, 0], [0, 3, 0], [0, 1, 1]]
    assert metricas["orden_clases"] == ["baja", "media", "alta"]


def test_sin_orden_la_matriz_es_alfabetica_y_no_inventa_el_campo():
    metricas = metricas_clasificacion(["b", "a"], ["b", "a"])
    assert metricas["matriz_confusion"] == [[1, 0], [0, 1]] and "orden_clases" not in metricas


@pytest.mark.parametrize("valor, fuera", [(5, []), (0, []), (-0.4, []), (-0.6, ["x"]), (10.4, []), (10.6, ["x"])])
def test_fuera_de_rango_usa_un_margen_del_5_por_ciento_del_ancho(tmp_path, valor, fuera):
    """Rango [0, 10]: el margen es 0.5, así que -0.5 y 10.5 aún se toleran."""
    joblib.dump({"rango": {"x": [0.0, 10.0]}}, tmp_path / "modelo.joblib")
    assert fuera_de_rango(tmp_path, "m", {"x": valor}) == fuera


def test_fuera_de_rango_sin_rango_guardado_no_avisa(tmp_path):
    joblib.dump({"pipeline": None}, tmp_path / "modelo.joblib")
    assert fuera_de_rango(tmp_path, "m", {"x": np.inf}) == []

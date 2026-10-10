"""Pruebas de las utilidades comunes de entrenamiento y predicción (backend/core)."""
import joblib
import numpy as np
import pytest

from core.entrenamiento import metricas_clasificacion, umbral_optimo_f1
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


def test_umbral_optimo_f1_baja_el_umbral_cuando_la_clase_positiva_es_rara():
    """Con probabilidades bien calibradas y una clase rara, el F1 se maximiza con un umbral menor que 0.5."""
    rng = np.random.default_rng(0)
    p = rng.beta(1, 6, 5000)                      # probabilidad de ~14 % en promedio
    y = (rng.random(5000) < p).astype(int)        # etiquetas coherentes con p
    assert umbral_optimo_f1(y, p) < 0.4


def test_umbral_optimo_f1_con_separacion_perfecta_y_empate():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    umbral = umbral_optimo_f1(y, p)
    assert 0.3 < umbral <= 0.7, "debe separar las clases"
    assert umbral == 0.31, "en caso de empate gana el umbral más bajo"


def test_umbral_optimo_f1_acepta_etiquetas_booleanas():
    assert umbral_optimo_f1([False, True, True], [0.1, 0.6, 0.9]) == pytest.approx(0.11)

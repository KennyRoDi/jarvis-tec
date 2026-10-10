"""Pruebas de las utilidades comunes de entrenamiento y predicción (backend/core)."""
import os

import joblib
import numpy as np
import pytest

from core.entrenamiento import intervalo_bootstrap, metricas_clasificacion, umbral_optimo_f1, umbral_para_recall
from core.modelos import cargar_artefacto, fuera_de_rango


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


def test_fuera_de_rango_revisa_todos_los_campos_y_los_devuelve(tmp_path):
    joblib.dump({"rango": {"a": [0.0, 10.0], "b": [0.0, 10.0], "c": [0.0, 10.0]}}, tmp_path / "modelo.joblib")
    assert fuera_de_rango(tmp_path, "m", {"a": 5, "b": 50, "c": -50}) == ["b", "c"]


def test_el_cache_del_artefacto_se_invalida_cuando_cambia_el_archivo(tmp_path):
    """Si se reentrena con el servidor encendido, debe servirse el artefacto nuevo."""
    ruta = tmp_path / "modelo.joblib"
    joblib.dump({"version": 1}, ruta)
    assert cargar_artefacto(tmp_path, "m")["version"] == 1
    joblib.dump({"version": 2}, ruta)
    os.utime(ruta, (ruta.stat().st_atime, ruta.stat().st_mtime + 10))
    assert cargar_artefacto(tmp_path, "m")["version"] == 2


def test_umbral_para_recall_devuelve_el_mayor_umbral_que_alcanza_el_objetivo():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.5, 0.3, 0.6, 0.2, 0.1, 0.05])
    assert umbral_para_recall(y, p, 0.5) == 0.7      # 2 de 4 positivos con p >= 0.7
    assert umbral_para_recall(y, p, 0.75) == 0.5     # 3 de 4 con p >= 0.5
    assert umbral_para_recall(y, p, 1.0) == 0.3      # los 4 solo con p >= 0.3


def test_umbral_para_recall_sin_positivos_alcanzables_devuelve_el_minimo():
    assert umbral_para_recall([1, 1], [0.0, 0.0], 0.9, rejilla=[0.1, 0.2, 0.3]) == 0.1


def test_intervalo_bootstrap_contiene_el_valor_y_se_ensancha_con_pocos_datos():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    def datos(n):
        y = (rng.random(n) < 0.3).astype(int)
        return y, np.clip(0.5 * y + rng.normal(0.3, 0.2, n), 0, 1)
    y1, p1 = datos(2000)
    y2, p2 = datos(60)
    ic_grande, ic_pequeno = intervalo_bootstrap(y1, p1, roc_auc_score), intervalo_bootstrap(y2, p2, roc_auc_score)
    assert ic_grande[0] < roc_auc_score(y1, p1) < ic_grande[1]
    assert (ic_pequeno[1] - ic_pequeno[0]) > 3 * (ic_grande[1] - ic_grande[0])
    assert intervalo_bootstrap(y1, p1, roc_auc_score) == ic_grande, "determinista con la misma semilla"

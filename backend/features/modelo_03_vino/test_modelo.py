"""Pruebas del modelo 03 · vino (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import numpy as np
import pandas as pd
import pytest

from core.modelos import REGISTRO, predecir_con_pipeline
from features.modelo_03_vino.router import CARPETA, ORDEN
from features.modelo_03_vino.train import OBJETIVO, VARIABLES, agrupar_calidad, cargar_datos

URL = "/api/modelos/vino/predecir"
ENTRADA = {
    "tipo": "white", "fixed_acidity": 7.0, "volatile_acidity": 0.3, "citric_acid": 0.32, "residual_sugar": 2.0,
    "chlorides": 0.045, "free_sulfur_dioxide": 30.0, "total_sulfur_dioxide": 115.0, "density": 0.994,
    "ph": 3.2, "sulphates": 0.5, "alcohol": 10.5,
}
entrenado = pytest.mark.skipif(not REGISTRO["vino"].entrenado, reason="ejecutar features.modelo_03_vino.train")


# --- Preparación de datos (no requiere el modelo entrenado) ---

@pytest.mark.parametrize("puntaje, clase", [(3, "baja"), (5, "baja"), (6, "media"), (7, "alta"), (9, "alta")])
def test_agrupacion_de_la_calidad(puntaje, clase):
    assert agrupar_calidad(pd.Series([puntaje])).iloc[0] == clase


def test_no_quedan_duplicados_antes_de_dividir():
    """Con duplicados, filas idénticas caen en entrenamiento y prueba: la exactitud sube ~10 puntos sin generalizar."""
    df = cargar_datos()
    assert not df.duplicated().any() and len(df) == 5329


def test_las_variables_no_incluyen_el_objetivo_ni_la_puntuacion():
    assert "quality" not in VARIABLES and OBJETIVO not in VARIABLES and len(VARIABLES) == 12


# --- Modelo y API ---

@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "vino" and cuerpo["prediccion"] in ORDEN
    assert set(cuerpo["probabilidades"]) == set(ORDEN)
    assert sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert cuerpo["prediccion"] == max(cuerpo["probabilidades"], key=cuerpo["probabilidades"].get)
    assert f"calidad {cuerpo['prediccion']}" in cuerpo["texto"] and "blanco" in cuerpo["texto"]


@entrenado
def test_el_pipeline_imputa_los_nulos():
    """El dataset trae nulos; el imputador del Pipeline debe permitir predecir con valores faltantes."""
    incompleta = {**ENTRADA, "ph": np.nan, "citric_acid": np.nan}
    clase, probs = predecir_con_pipeline(CARPETA, "vino", incompleta)
    assert clase in ORDEN and sum(probs.values()) == pytest.approx(1, abs=0.01)


@entrenado
def test_mas_alcohol_aumenta_la_probabilidad_de_calidad_alta():
    """Coherencia con la exploración: el alcohol es la variable más asociada a la calidad (r = 0.47)."""
    filas = cargar_datos().sample(40, random_state=0)[VARIABLES].to_dict("records")
    def p_alta(fila, alcohol):
        return predecir_con_pipeline(CARPETA, "vino", {**fila, "alcohol": alcohol})[1]["alta"]
    mejora = np.mean([p_alta(f, min(f["alcohol"] + 2, 15)) > p_alta(f, max(f["alcohol"] - 2, 8)) for f in filas])
    assert mejora >= 0.9


@entrenado
def test_un_vino_de_perfil_malo_no_se_clasifica_como_alto():
    malo = {**ENTRADA, "alcohol": 8.5, "volatile_acidity": 0.9, "density": 0.999, "chlorides": 0.15}
    buena = {**ENTRADA, "alcohol": 13.0, "volatile_acidity": 0.2, "density": 0.990, "chlorides": 0.03}
    assert predecir_con_pipeline(CARPETA, "vino", malo)[1]["alta"] < predecir_con_pipeline(CARPETA, "vino", buena)[1]["alta"]


@pytest.mark.parametrize("cambio", [{"tipo": "rosado"}, {"alcohol": 40}, {"ph": 14}, {"density": -1}, {"alcohol": "mucho"}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_faltan_campos_da_422(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "alcohol"}), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_matriz_y_esquema(cliente):
    cuerpo = cliente.get("/api/modelos/vino/info").json()
    assert {"accuracy", "f1_macro", "precision_macro", "recall_macro"} <= set(cuerpo["metricas"])
    matriz = np.array(cuerpo["metricas"]["matriz_confusion"])
    assert matriz.shape == (3, 3) and matriz.sum() == cuerpo["metricas"]["n_prueba"]
    assert cuerpo["metricas"]["orden_clases"] == list(ORDEN)
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"])
    assert cuerpo["esquema_entrada"]["properties"]["tipo"]["enum"] == ["red", "white"]
    assert set(ENTRADA) == set(cuerpo["entrada_ejemplo"])


@entrenado
def test_el_modelo_supera_claramente_a_la_linea_base():
    """Guardia de regresión: si el entrenamiento se rompe, el F1 cae hacia el 0.20 de la clase mayoritaria."""
    from core.modelos import leer_metricas
    m = leer_metricas(CARPETA)["metricas"]
    assert m["f1_macro"] > m["baseline"]["f1_macro"] + 0.3


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000

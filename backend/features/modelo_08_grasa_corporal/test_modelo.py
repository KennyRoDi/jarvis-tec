"""Pruebas del modelo 08 · grasa corporal (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import pytest

from core.modelos import REGISTRO, predecir_con_pipeline
from features.modelo_08_grasa_corporal.router import CARPETA

ENTRADA = {
    "age": 45, "weight_kg": 80.0, "height_cm": 178.0, "neck_cm": 38.0, "chest_cm": 100.0, "abdomen_cm": 92.0,
    "hip_cm": 99.0, "thigh_cm": 59.0, "knee_cm": 38.5, "ankle_cm": 23.0, "biceps_cm": 32.0,
    "forearm_cm": 28.7, "wrist_cm": 18.3,
}
ENTRADA_EXTREMA = {**ENTRADA, "weight_kg": 165.0, "chest_cm": 136.0, "abdomen_cm": 148.0, "hip_cm": 147.0, "thigh_cm": 87.0}
entrenado = pytest.mark.skipif(not REGISTRO["grasa_corporal"].entrenado,
                               reason="ejecutar features.modelo_08_grasa_corporal.train")


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post("/api/modelos/grasa_corporal/predecir", json=ENTRADA).json()
    assert cuerpo["modelo"] == "grasa_corporal"
    assert 5 < cuerpo["prediccion"] < 35  # persona de medidas típicas: rango plausible
    assert cuerpo["unidad"] == "%"
    assert "poco confiable" not in cuerpo["texto"]


@entrenado
def test_prediccion_negativa_del_modelo_se_recorta_a_cero(cliente):
    """La esquina inferior del rango válido hace que Lasso prediga un valor negativo: la API debe devolver 0."""
    minimo = {**ENTRADA, "age": 18, "weight_kg": 45, "abdomen_cm": 65, "chest_cm": 75, "hip_cm": 80}
    crudo, _ = predecir_con_pipeline(CARPETA, "grasa_corporal", minimo)
    assert crudo < 0, "el caso ya no ejercita el recorte: elegir otra entrada"
    assert cliente.post("/api/modelos/grasa_corporal/predecir", json=minimo).json()["prediccion"] == 0.0


@entrenado
def test_medidas_extremas_avisan_que_el_resultado_es_poco_confiable(cliente):
    cuerpo = cliente.post("/api/modelos/grasa_corporal/predecir", json=ENTRADA_EXTREMA).json()
    assert "poco confiable" in cuerpo["texto"]


@entrenado
def test_abdomen_mayor_implica_mas_grasa(cliente):
    """Coherencia con lo explorado: el abdomen es la variable más correlacionada con la grasa."""
    menor = cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "abdomen_cm": 80}).json()["prediccion"]
    mayor = cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "abdomen_cm": 105}).json()["prediccion"]
    assert mayor > menor


def test_predecir_valida_entrada(cliente, assert_error):
    assert_error(cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "age": 5}), 422, "VALIDACION")


def test_predecir_exige_todos_los_campos(cliente, assert_error):
    incompleta = {k: v for k, v in ENTRADA.items() if k != "wrist_cm"}
    assert_error(cliente.post("/api/modelos/grasa_corporal/predecir", json=incompleta), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_y_esquema(cliente):
    cuerpo = cliente.get("/api/modelos/grasa_corporal/info").json()
    assert {"r2", "mae", "rmse"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"])
    assert set(ENTRADA) == set(cuerpo["entrada_ejemplo"])
    assert isinstance(cuerpo["entrada_ejemplo"]["age"], int)  # el formulario recibe un entero, no 23.0

"""Pruebas del modelo 09 · aguacate (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import pandas as pd
import pytest

from core.modelos import REGISTRO
from features.modelo_09_aguacate.router import CARPETA, REGIONES, nombre_region

URL = "/api/modelos/aguacate/predecir"
ENTRADA = {"region": "TotalUS", "tipo": "conventional", "fecha": "2017-09-15"}  # dentro del rango del dataset
entrenado = pytest.mark.skipif(not REGISTRO["aguacate"].entrenado, reason="ejecutar features.modelo_09_aguacate.train")


def precio(cliente, **cambios) -> float:
    return cliente.post(URL, json={**ENTRADA, **cambios}).json()["prediccion"]


def test_las_regiones_del_router_coinciden_con_el_dataset():
    del_dataset = set(pd.read_csv(CARPETA / "dataset.csv", usecols=["region"]).region)
    assert set(REGIONES) == del_dataset and len(REGIONES) == 54


def test_nombre_legible_de_la_region():
    assert nombre_region("LosAngeles") == "Los Angeles"
    assert nombre_region("TotalUS") == "todo Estados Unidos"
    assert nombre_region("DallasFtWorth") == "Dallas Ft. Worth"


@entrenado
def test_predecir_dentro_del_rango_no_avisa(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "aguacate" and cuerpo["unidad"] == "USD por aguacate"
    assert 0.5 < cuerpo["prediccion"] < 3.5
    assert "15 de septiembre de 2017" in cuerpo["texto"] and "poco confiable" not in cuerpo["texto"]


@entrenado
def test_solicitud_vacia_es_valida_y_usa_los_valores_por_defecto(cliente):
    """Permite ejecutarlo solo con la voz ("precio del aguacate"). Hoy está lejos de los datos: debe avisar."""
    respuesta = cliente.post(URL, json={})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "todo Estados Unidos" in cuerpo["texto"] and "convencional" in cuerpo["texto"]
    assert "poco confiable" in cuerpo["texto"]


@entrenado
def test_el_organico_cuesta_mas_que_el_convencional(cliente):
    assert precio(cliente, tipo="organic") > precio(cliente, tipo="conventional")


@entrenado
def test_las_regiones_se_distinguen(cliente):
    """San Francisco es una de las regiones más caras; Houston, de las más baratas (ver figuras/regiones_extremas.png)."""
    assert precio(cliente, region="SanFrancisco") > precio(cliente, region="Houston")


@entrenado
def test_septiembre_es_mas_caro_que_febrero(cliente):
    """Estacionalidad observada en la exploración: el precio medio sube hacia septiembre-octubre."""
    assert precio(cliente, fecha="2017-09-15") > precio(cliente, fecha="2017-02-15")


@entrenado
def test_fecha_futura_avisa_y_sigue_respondiendo(cliente):
    cuerpo = cliente.post(URL, json={**ENTRADA, "fecha": "2030-06-01"}).json()
    assert cuerpo["prediccion"] > 0 and "poco confiable" in cuerpo["texto"]


@pytest.mark.parametrize("cambio", [{"region": "Narnia"}, {"tipo": "frito"}, {"fecha": "ayer"}, {"fecha": "1999-01-01"}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_y_esquema_con_las_54_regiones(cliente):
    cuerpo = cliente.get("/api/modelos/aguacate/info").json()
    assert {"r2", "mae", "rmse"} <= set(cuerpo["metricas"])
    assert len(cuerpo["esquema_entrada"]["properties"]["region"]["enum"]) == 54
    assert not cuerpo["esquema_entrada"].get("required")  # todos con valor por defecto


@entrenado
def test_el_artefacto_no_es_excesivo():
    """El joblib se versiona en git: vigilar que no crezca sin control."""
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000

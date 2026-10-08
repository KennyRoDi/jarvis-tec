"""Pruebas del modelo 02 · autos (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import pytest

from core.modelos import REGISTRO

ENTRADA = {
    "year": 2014, "present_price": 5.59, "kms_driven": 27000, "fuel_type": "Petrol",
    "seller_type": "Dealer", "transmission": "Manual", "owner": 0,
}
entrenado = pytest.mark.skipif(not REGISTRO["autos"].entrenado, reason="ejecutar features.modelo_02_autos.train")


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post("/api/modelos/autos/predecir", json=ENTRADA).json()
    assert cuerpo["modelo"] == "autos"
    assert cuerpo["prediccion"] > 0
    assert cuerpo["unidad"] == "lakhs INR"
    assert cuerpo["texto"]


def test_predecir_valida_entrada(cliente, assert_error):
    assert_error(cliente.post("/api/modelos/autos/predecir", json={**ENTRADA, "fuel_type": "Agua"}), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas(cliente):
    assert "r2" in cliente.get("/api/modelos/autos/info").json()["metricas"]


def test_info_incluye_esquema_de_entrada(cliente):
    esquema = cliente.get("/api/modelos/autos/info").json()["esquema_entrada"]
    assert set(ENTRADA) == set(esquema["properties"])
    assert esquema["properties"]["fuel_type"]["enum"] == ["Petrol", "Diesel", "CNG"]

"""Pruebas del modelo 05 · acv (specs/api_rest_spec.md §3). Usan el modelo entrenado real.

Plantilla: completar según SPEC.md (ejemplo completo: features/modelo_02_autos/test_modelo.py).
"""
import pytest

from core.modelos import REGISTRO

ENTRADA: dict = {}  # TODO(Dev A): entrada válida según la clase Entrada de router.py
entrenado = pytest.mark.skipif(not REGISTRO["acv"].entrenado, reason="ejecutar features.modelo_05_acv.train")


@entrenado
@pytest.mark.skip(reason="TODO(Dev A): definir ENTRADA")
def test_predecir(cliente):
    cuerpo = cliente.post("/api/modelos/acv/predecir", json=ENTRADA).json()
    assert cuerpo["modelo"] == "acv"
    assert cuerpo["texto"]


@pytest.mark.skip(reason="TODO(Dev A): enviar una entrada inválida y esperar 422")
def test_predecir_valida_entrada(cliente, assert_error):
    assert_error(cliente.post("/api/modelos/acv/predecir", json={}), 422, "VALIDACION")

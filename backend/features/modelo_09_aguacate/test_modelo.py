"""Pruebas del modelo 09 · aguacate (specs/api_rest_spec.md §3). Usan el modelo entrenado real.

Plantilla: completar según SPEC.md (ejemplo completo: features/modelo_02_autos/test_modelo.py).
"""
import pytest

from core.modelos import REGISTRO

ENTRADA: dict = {}  # TODO(Dev A): entrada válida según la clase Entrada de router.py
entrenado = pytest.mark.skipif(not REGISTRO["aguacate"].entrenado, reason="ejecutar features.modelo_09_aguacate.train")


@entrenado
@pytest.mark.skip(reason="TODO(Dev A): definir ENTRADA")
def test_predecir(cliente):
    cuerpo = cliente.post("/api/modelos/aguacate/predecir", json=ENTRADA).json()
    assert cuerpo["modelo"] == "aguacate"
    assert cuerpo["texto"]


@pytest.mark.skip(reason="TODO(Dev A): enviar una entrada inválida y esperar 422")
def test_predecir_valida_entrada(cliente, assert_error):
    assert_error(cliente.post("/api/modelos/aguacate/predecir", json={}), 422, "VALIDACION")

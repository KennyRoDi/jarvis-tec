"""Pruebas transversales del contrato (specs/api_rest_spec.md §1–§3).

Las pruebas de cada feature viven en su carpeta: features/<feature>/test_*.py.
Ejecutar:  cd backend && pytest
"""
import pytest

from core.modelos import REGISTRO
from features.asistente_voz.comandos import interpretar


def test_salud(cliente):
    assert cliente.get("/api/salud").json()["estado"] == "ok"


def test_raiz_sirve_la_interfaz(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert "JarvisTEC" in respuesta.text


def test_se_registran_los_10_modelos(cliente):
    modelos = cliente.get("/api/modelos").json()["modelos"]
    assert len(modelos) == 10
    assert {"id", "slug", "nombre", "tipo", "comandos", "entrenado"} <= set(modelos[0])


def test_ruta_inexistente_usa_formato_de_error(cliente, assert_error):
    assert_error(cliente.get("/api/no-existe"), 404, "NO_ENCONTRADO")


def test_json_con_nan_es_un_422_no_un_500(cliente, assert_error):
    """NaN no es JSON estándar, pero Python lo acepta: antes provocaba un 500 al serializar el error."""
    respuesta = cliente.post("/api/modelos/autos/predecir", content=b'{"year": NaN}',
                             headers={"Content-Type": "application/json"})
    assert_error(respuesta, 422, "VALIDACION")
    assert {"loc", "msg", "type"} <= set(respuesta.json()["error"]["detalle"][0])


def test_info_de_modelo_inexistente(cliente, assert_error):
    assert_error(cliente.get("/api/modelos/no-existe/info"), 404, "NO_ENCONTRADO")


def test_modelo_no_entrenado_responde_503(cliente, assert_error):
    pendientes = [m for m in REGISTRO.values() if not m.entrenado]
    if not pendientes:
        pytest.skip("todos los modelos están entrenados")
    respuesta = cliente.post(f"/api/modelos/{pendientes[0].info['slug']}/predecir", json={})
    assert_error(respuesta, 503, "MODELO_NO_ENTRENADO")


def test_cada_comando_de_voz_se_asocia_a_su_propio_modelo():
    """Los comandos no deben repetirse ni confundirse entre modelos (el intérprete elige la frase más larga)."""
    for slug, modelo in REGISTRO.items():
        for comando in modelo.info["comandos"]:
            assert interpretar(comando)["modelo"] == slug, f"'{comando}' ({slug}) se asocia a otro modelo"

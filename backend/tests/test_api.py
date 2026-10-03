"""Pruebas del contrato specs/api_rest_spec.md.  Ejecutar:  cd backend && pytest"""
import pytest
from fastapi.testclient import TestClient

from core.modelos import REGISTRO
from main import app

cliente = TestClient(app)

ENTRADA_AUTO = {
    "year": 2014, "present_price": 5.59, "kms_driven": 27000, "fuel_type": "Petrol",
    "seller_type": "Dealer", "transmission": "Manual", "owner": 0,
}


def assert_error(respuesta, status: int, codigo: str):
    assert respuesta.status_code == status
    error = respuesta.json()["error"]
    assert error["codigo"] == codigo
    assert error["mensaje"]


def test_salud():
    assert cliente.get("/api/salud").json()["estado"] == "ok"


def test_se_registran_los_10_modelos():
    modelos = cliente.get("/api/modelos").json()["modelos"]
    assert len(modelos) == 10
    assert {"id", "slug", "nombre", "tipo", "comandos", "entrenado"} <= set(modelos[0])


def test_ruta_inexistente_usa_formato_de_error():
    assert_error(cliente.get("/api/no-existe"), 404, "NO_ENCONTRADO")


def test_info_de_modelo_inexistente():
    assert_error(cliente.get("/api/modelos/no-existe/info"), 404, "NO_ENCONTRADO")


def test_modelo_no_entrenado_responde_503():
    pendientes = [m for m in REGISTRO.values() if not m.entrenado]
    if not pendientes:
        pytest.skip("todos los modelos están entrenados")
    assert_error(cliente.post(f"/api/modelos/{pendientes[0].info['slug']}/predecir", json={}), 503,
                 "MODELO_NO_ENTRENADO")


@pytest.mark.skipif(not REGISTRO["autos"].entrenado, reason="ejecutar features.modelo_02_autos.train")
def test_predecir_autos():
    cuerpo = cliente.post("/api/modelos/autos/predecir", json=ENTRADA_AUTO).json()
    assert cuerpo["modelo"] == "autos"
    assert cuerpo["prediccion"] > 0
    assert cuerpo["unidad"] == "lakhs INR"
    assert cuerpo["texto"]


def test_predecir_autos_valida_entrada():
    respuesta = cliente.post("/api/modelos/autos/predecir", json={**ENTRADA_AUTO, "fuel_type": "Agua"})
    assert_error(respuesta, 422, "VALIDACION")


def test_info_autos_incluye_metricas():
    cuerpo = cliente.get("/api/modelos/autos/info").json()
    if cuerpo["entrenado"]:
        assert "r2" in cuerpo["metricas"]


@pytest.mark.parametrize("texto, esperado", [
    ("JarvisTEC precio del bitcoin para mañana", "bitcoin"),
    ("Jarvis, ¿cuánto vale mi carro?", "autos"),
    ("cuál es la calidad del vino", "vino"),
])
def test_comando_reconocido(texto, esperado):
    cuerpo = cliente.post("/api/asistente/comando", json={"texto": texto}).json()
    assert cuerpo["reconocido"] is True
    assert cuerpo["modelo"] == esperado


def test_comando_no_reconocido():
    cuerpo = cliente.post("/api/asistente/comando", json={"texto": "Jarvis cuéntame un chiste"}).json()
    assert cuerpo == {**cuerpo, "reconocido": False, "modelo": None}


def test_transcribir_rechaza_formato():
    respuesta = cliente.post("/api/voz/transcribir", files={"audio": ("a.txt", b"hola", "text/plain")})
    assert_error(respuesta, 415, "TIPO_NO_SOPORTADO")


def test_transcribir_rechaza_audio_vacio():
    respuesta = cliente.post("/api/voz/transcribir", files={"audio": ("a.webm", b"", "audio/webm")})
    assert_error(respuesta, 400, "SOLICITUD_INVALIDA")


def test_emocion_rechaza_imagen_vacia():
    respuesta = cliente.post("/api/vision/emocion", files={"imagen": ("a.jpg", b"", "image/jpeg")})
    assert_error(respuesta, 400, "SOLICITUD_INVALIDA")


def test_raiz_sirve_la_interfaz():
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert "JarvisTEC" in respuesta.text

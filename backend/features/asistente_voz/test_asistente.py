"""Pruebas del asistente de voz (specs/api_rest_spec.md §4)."""
import pytest


@pytest.mark.parametrize("texto, esperado", [
    ("JarvisTEC precio del bitcoin para mañana", "bitcoin"),
    ("Jarvis, ¿cuánto vale mi carro?", "autos"),
    ("cuál es la calidad del vino", "vino"),
])
def test_comando_reconocido(cliente, texto, esperado):
    cuerpo = cliente.post("/api/asistente/comando", json={"texto": texto}).json()
    assert cuerpo["reconocido"] is True
    assert cuerpo["modelo"] == esperado


def test_comando_no_reconocido(cliente):
    cuerpo = cliente.post("/api/asistente/comando", json={"texto": "Jarvis cuéntame un chiste"}).json()
    assert cuerpo["reconocido"] is False
    assert cuerpo["modelo"] is None


def test_comando_devuelve_la_emocion_recibida(cliente):
    cuerpo = cliente.post("/api/asistente/comando", json={"texto": "precio del bitcoin", "emocion": "tristeza"}).json()
    assert cuerpo["emocion"] == "tristeza"


def test_comando_rechaza_emocion_desconocida(cliente, assert_error):
    respuesta = cliente.post("/api/asistente/comando", json={"texto": "precio del bitcoin", "emocion": "aburrido"})
    assert_error(respuesta, 422, "VALIDACION")


def test_transcribir_rechaza_formato(cliente, assert_error):
    respuesta = cliente.post("/api/voz/transcribir", files={"audio": ("a.txt", b"hola", "text/plain")})
    assert_error(respuesta, 415, "TIPO_NO_SOPORTADO")


def test_transcribir_rechaza_audio_vacio(cliente, assert_error):
    respuesta = cliente.post("/api/voz/transcribir", files={"audio": ("a.webm", b"", "audio/webm")})
    assert_error(respuesta, 400, "SOLICITUD_INVALIDA")

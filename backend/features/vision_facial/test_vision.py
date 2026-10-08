"""Pruebas de visión facial (specs/api_rest_spec.md §5)."""


def test_emocion_rechaza_formato(cliente, assert_error):
    respuesta = cliente.post("/api/vision/emocion", files={"imagen": ("a.gif", b"GIF89a", "image/gif")})
    assert_error(respuesta, 415, "TIPO_NO_SOPORTADO")


def test_emocion_rechaza_imagen_vacia(cliente, assert_error):
    respuesta = cliente.post("/api/vision/emocion", files={"imagen": ("a.jpg", b"", "image/jpeg")})
    assert_error(respuesta, 400, "SOLICITUD_INVALIDA")

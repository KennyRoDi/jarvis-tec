"""Emoción del rostro con Google Cloud Vision `face_detection` (Dev B). Ver SPEC.md, opción 1.

Requisitos: GOOGLE_APPLICATION_CREDENTIALS en backend/.env (la misma cuenta de servicio que Speech-to-Text,
con la API de Cloud Vision habilitada en el proyecto de GCP).
"""
from core.errores import ApiError

# Probabilidad cualitativa de Vision -> puntaje del contrato (§5)
PUNTAJE_POR_PROBABILIDAD = {
    "UNKNOWN": 0.0, "VERY_UNLIKELY": 0.0, "UNLIKELY": 0.25, "POSSIBLE": 0.5, "LIKELY": 0.75, "VERY_LIKELY": 1.0,
}
# Atributo de FaceAnnotation -> clave del contrato
EMOCION_POR_ATRIBUTO = {
    "joy_likelihood": "felicidad", "sorrow_likelihood": "tristeza",
    "anger_likelihood": "enojo", "surprise_likelihood": "sorpresa",
}


def puntajes_de_rostro(contenido: bytes) -> list[dict[str, float]]:
    """Devuelve un dict de puntajes (claves de EMOCION_POR_ATRIBUTO) por cada rostro, de izquierda a derecha."""
    # TODO(Dev B): vision.ImageAnnotatorClient().face_detection(image=vision.Image(content=contenido))
    #   - Revisar respuesta.error.message: si trae texto, ApiError(502, "SERVICIO_EXTERNO", ...)
    #   - Ordenar los rostros por posición para emparejarlos con los de Azure
    raise ApiError(501, "NO_IMPLEMENTADO", "La emoción con Google Vision aún no está implementada.")

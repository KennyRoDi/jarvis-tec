"""Integración con Azure AI Face (Dev B).

Requisitos: AZURE_FACE_ENDPOINT y AZURE_FACE_KEY en backend/.env. SDK: azure-cognitiveservices-vision-face.

⚠️ Microsoft retiró el atributo `emotion` de Face API (anunciado en junio 2022, acceso cerrado para
recursos nuevos). Antes de implementar, verificar con una llamada real si el recurso del equipo lo
devuelve; si no, consultar al profesor por una alternativa (el enunciado permite APIs de Google para
sentimientos en esta entrega, p. ej. Google Cloud Vision `face_detection`). El contrato de §5 no cambia.
"""
from core.errores import ApiError

EMOCIONES = ("felicidad", "tristeza", "enojo", "sorpresa", "miedo", "desprecio", "disgusto", "neutral")


def detectar_emociones(contenido: bytes) -> list[dict]:
    """Devuelve la lista `rostros` de specs/api_rest_spec.md §5 (puntajes con las 8 claves de EMOCIONES)."""
    # TODO(Dev B): FaceClient(endpoint, CognitiveServicesCredentials(key)).face.detect_with_stream(...) y mapear la respuesta.
    #   Envolver errores de azure.cognitiveservices.vision.face.models.APIErrorException en ApiError(502, "SERVICIO_EXTERNO", ...)
    raise ApiError(501, "NO_IMPLEMENTADO", "La detección de emociones aún no está implementada.")

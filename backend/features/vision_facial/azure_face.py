"""Detección de rostros con Azure Face + emoción con Google Vision (Dev B). Ver SPEC.md, opción 1.

Requisitos: AZURE_FACE_ENDPOINT y AZURE_FACE_KEY en backend/.env. SDK: azure-cognitiveservices-vision-face.
Azure aporta el `rectangulo` (detección sin atributos de emoción, que Microsoft retiró en 2022);
`google_vision.puntajes_de_rostro` aporta los puntajes. El contrato §5 no depende del proveedor.
"""
from core.errores import ApiError

EMOCIONES = ("felicidad", "tristeza", "enojo", "sorpresa", "miedo", "desprecio", "disgusto", "neutral")


def detectar_emociones(contenido: bytes) -> list[dict]:
    """Devuelve la lista `rostros` de specs/api_rest_spec.md §5 (puntajes con las 8 claves de EMOCIONES)."""
    # TODO(Dev B):
    #   1. Azure: FaceClient(endpoint, CognitiveServicesCredentials(key)).face.detect_with_stream(imagen,
    #      return_face_attributes=None, detection_model="detection_03") -> face_rectangle de cada rostro.
    #   2. Google: google_vision.puntajes_de_rostro(contenido); emparejar con Azure por posición.
    #   3. Completar las 8 claves de EMOCIONES (las que Vision no da = 0.0; neutral = 1 - máximo).
    #   Envolver errores de azure.cognitiveservices.vision.face.models.APIErrorException en ApiError(502, "SERVICIO_EXTERNO", ...)
    raise ApiError(501, "NO_IMPLEMENTADO", "La detección de emociones aún no está implementada.")

"""Integración con Google Cloud Speech-to-Text (Dev B).

Requisitos: GOOGLE_APPLICATION_CREDENTIALS en backend/.env apuntando al JSON de la cuenta de servicio.
Guía: https://cloud.google.com/speech-to-text/docs/sync-recognize
"""
from core.errores import ApiError


def transcribir_audio(contenido: bytes, tipo: str, idioma: str) -> dict:
    """Devuelve {"texto", "confianza", "idioma"} según specs/api_rest_spec.md §4."""
    # TODO(Dev B): implementar con google.cloud.speech.SpeechClient().recognize(...).
    #   - audio/webm (MediaRecorder del navegador) -> RecognitionConfig.AudioEncoding.WEBM_OPUS, 48000 Hz
    #   - audio/wav -> LINEAR16 (la frecuencia se lee del encabezado)
    #   - Envolver errores de google.api_core.exceptions en ApiError(502, "SERVICIO_EXTERNO", ...)
    raise ApiError(501, "NO_IMPLEMENTADO", "La transcripción de voz aún no está implementada.")

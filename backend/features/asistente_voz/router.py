"""Feature: asistente de voz (Dev B). Contrato: specs/api_rest_spec.md §4."""
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from core.errores import ApiError
from features.asistente_voz.comandos import interpretar
from features.asistente_voz.speech import transcribir_audio

router = APIRouter(tags=["Asistente de voz"])

TIPOS_AUDIO = ("audio/webm", "audio/wav", "audio/x-wav", "audio/wave", "audio/ogg")


Emocion = Literal["felicidad", "tristeza", "enojo", "sorpresa", "miedo", "desprecio", "disgusto", "neutral"]


class Comando(BaseModel):
    texto: str = Field(min_length=1, examples=["JarvisTEC precio del bitcoin para mañana"])
    emocion: Emocion | None = Field(None, description="Última emoción dominante detectada (§5)")


@router.post("/api/voz/transcribir")
async def transcribir(audio: UploadFile = File(...), idioma: str = Form("es-CR")):
    tipo = (audio.content_type or "").split(";")[0].strip()
    if tipo not in TIPOS_AUDIO:
        raise ApiError(415, "TIPO_NO_SOPORTADO", f"Formato de audio no soportado: '{tipo}'.")
    contenido = await audio.read()
    if not contenido:
        raise ApiError(400, "SOLICITUD_INVALIDA", "El archivo de audio está vacío.")
    return transcribir_audio(contenido, tipo, idioma)


@router.post("/api/asistente/comando")
def comando(cuerpo: Comando):
    return interpretar(cuerpo.texto, cuerpo.emocion)

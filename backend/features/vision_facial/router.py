"""Feature: visión facial (Dev B). Contrato: specs/api_rest_spec.md §5."""
from fastapi import APIRouter, File, UploadFile

from core.errores import ApiError
from features.vision_facial.azure_face import detectar_emociones

router = APIRouter(prefix="/api/vision", tags=["Visión facial"])

TIPOS_IMAGEN = ("image/jpeg", "image/png")


@router.post("/emocion")
async def emocion(imagen: UploadFile = File(...)):
    if imagen.content_type not in TIPOS_IMAGEN:
        raise ApiError(415, "TIPO_NO_SOPORTADO", f"Formato de imagen no soportado: '{imagen.content_type}'.")
    contenido = await imagen.read()
    if not contenido:
        raise ApiError(400, "SOLICITUD_INVALIDA", "La imagen está vacía.")
    rostros = detectar_emociones(contenido)
    return {"cantidad": len(rostros), "rostros": rostros}

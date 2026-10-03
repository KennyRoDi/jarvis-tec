"""Modelo 01 · Predicción del precio del Bitcoin.

Contrato: specs/api_rest_spec.md §3 · Esquema de entrada: specs/modelos_spec.md (01 · bitcoin).
Referencia completa: features/modelo_02_autos/router.py
"""
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from core.modelos import RespuestaPrediccion, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "bitcoin",
    "nombre": "Predicción del precio del Bitcoin",
    "tipo": "regresion",
    "unidad": 'USD',
    "comandos": ["precio del bitcoin", "bitcoin manana", "tipo de cambio del bitcoin"],
}

router = APIRouter(prefix="/api/modelos/bitcoin", tags=["Modelo 01 · Bitcoin"])


class Entrada(BaseModel):
    # TODO(Dev A): declarar los campos de entrada (mismos nombres que las columnas usadas en train.py),
    # documentarlos en specs/modelos_spec.md y eliminar extra="allow".
    model_config = ConfigDict(extra="allow")


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    valor, probabilidades = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], entrada.model_dump())
    return RespuestaPrediccion(
        modelo=MODELO_INFO["slug"],
        prediccion=valor,
        unidad=MODELO_INFO["unidad"],
        probabilidades=probabilidades,
        texto=f"El resultado del modelo es {valor}.",  # TODO(Dev A): frase natural para JARVIS
    )

"""Modelo 07 · Clasificación de la etapa de cirrosis.

Contrato: specs/api_rest_spec.md §3 · Esquema de entrada: specs/modelos_spec.md (07 · cirrosis).
Referencia completa: features/modelo_02_autos/router.py
"""
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from core.modelos import RespuestaPrediccion, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "cirrosis",
    "nombre": "Clasificación de la etapa de cirrosis",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["etapa de cirrosis", "tipo de cirrosis"],
}

router = APIRouter(prefix="/api/modelos/cirrosis", tags=["Modelo 07 · Cirrosis"])


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

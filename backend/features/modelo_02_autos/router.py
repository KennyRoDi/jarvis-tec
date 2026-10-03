"""Modelo 02 · Predicción del precio de un automóvil usado.

Contrato: specs/api_rest_spec.md §3 · Esquema de entrada: specs/modelos_spec.md (02 · autos).
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.modelos import RespuestaPrediccion, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "autos",
    "nombre": "Predicción del precio de un automóvil",
    "tipo": "regresion",
    "unidad": "lakhs INR",
    "comandos": ["precio de un auto", "precio de un carro", "cuanto vale mi carro"],
}

router = APIRouter(prefix="/api/modelos/autos", tags=["Modelo 02 · Autos"])


class Entrada(BaseModel):
    year: int = Field(ge=1990, le=2030, examples=[2014])
    present_price: float = Field(gt=0, description="Precio de agencia actual (lakhs INR)", examples=[5.59])
    kms_driven: int = Field(ge=0, examples=[27000])
    fuel_type: Literal["Petrol", "Diesel", "CNG"]
    seller_type: Literal["Dealer", "Individual"]
    transmission: Literal["Manual", "Automatic"]
    owner: int = Field(ge=0, le=3, description="Cantidad de dueños anteriores", examples=[0])


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    valor, _ = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], entrada.model_dump())
    valor = round(valor, 2)
    return RespuestaPrediccion(
        modelo=MODELO_INFO["slug"],
        prediccion=valor,
        unidad=MODELO_INFO["unidad"],
        texto=f"El precio estimado del vehículo es {valor} {MODELO_INFO['unidad']}.",
    )

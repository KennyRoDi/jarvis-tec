"""Modelo 02 · Predicción del precio de un automóvil usado.

Contrato: specs/api_rest_spec.md §3 · Esquema de entrada: specs/modelos_spec.md (02 · autos).
El precio de reventa se da en lakhs de rupias indias (1 lakh = 100 000 INR). El modelo aprende la razón entre la reventa y el precio de
agencia actual (ver `razon.py`), por lo que el precio nunca supera al de agencia.
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "autos",
    "nombre": "Predicción del precio de un automóvil",
    "tipo": "regresion",
    "unidad": "lakhs INR",
    "comandos": ["precio de un auto", "precio de un carro", "cuanto vale mi carro"],
}

router = APIRouter(prefix="/api/modelos/autos", tags=["Modelo 02 · Autos"])

RUPIAS_POR_LAKH = 100_000
DUENOS_RAROS = 2  # con 2 o más dueños anteriores hay un solo auto en los datos (con 3)


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos ("2014", true, 2014.5 en el año).
    model_config = ConfigDict(extra="forbid", strict=True)

    year: int = Field(ge=1990, le=2026, description="Año de fabricación", examples=[2014])
    present_price: float = Field(gt=0, le=150, description="Precio de agencia actual (lakhs INR)", examples=[5.59])
    kms_driven: int = Field(ge=0, le=1_000_000, description="Kilometraje recorrido (km)", examples=[27000])
    fuel_type: Literal["Petrol", "Diesel", "CNG"] = Field(description="Combustible: gasolina, diésel o gas natural", examples=["Petrol"])
    seller_type: Literal["Dealer", "Individual"] = Field(description="Vendedor: agencia (Dealer) o particular (Individual)", examples=["Dealer"])
    transmission: Literal["Manual", "Automatic"] = Field(description="Transmisión manual o automática", examples=["Manual"])
    owner: int = Field(ge=0, le=3, description="Cantidad de dueños anteriores", examples=[0])


def formato(numero: float) -> str:
    return f"{numero:,.0f}".replace(",", " ")


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    valor, _ = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    valor = round(max(valor, 0.0), 2)
    porcentaje = round(valor / entrada.present_price * 100)
    texto = (f"El precio de reventa estimado del vehículo es de {valor} lakhs de rupias indias, unas {formato(valor * RUPIAS_POR_LAKH)} rupias, "
             f"alrededor del {porcentaje} por ciento de su precio de agencia.")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: el año, el kilometraje o el precio de agencia está fuera del rango de los autos con que se entrenó el modelo y el resultado es poco confiable."
    if entrada.fuel_type == "CNG" or entrada.owner >= DUENOS_RAROS:
        texto += " Atención: hay muy pocos autos de gas natural o con varios dueños en los datos, así que la estimación es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=valor, unidad=MODELO_INFO["unidad"], texto=texto)

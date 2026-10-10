"""Modelo 01 · Predicción del precio del Bitcoin.

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
Todos los campos tienen valor por defecto: `{}` es una solicitud válida (así se ejecuta solo con la voz).
El dataset termina el 31-jul-2017: el modelo no conoce el precio actual y la respuesta lo dice. Para varios días se
encadenan predicciones de un día (ver `serie.recursiva`). El intervalo del 95 % crece con la raíz del número de días.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, cargar_artefacto
from features.modelo_01_bitcoin.serie import recursiva

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "bitcoin",
    "nombre": "Predicción del precio del Bitcoin",
    "tipo": "regresion",
    "unidad": "USD",
    "comandos": ["precio del bitcoin", "bitcoin manana", "tipo de cambio del bitcoin"],
}

router = APIRouter(prefix="/api/modelos/bitcoin", tags=["Modelo 01 · Bitcoin"])

MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre")
MAX_DIAS = 7
Z95 = 1.96


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos ("3", 2.5, true).
    model_config = ConfigDict(extra="forbid", strict=True)

    dias_adelante: int = Field(1, ge=1, le=MAX_DIAS, description="Días después del último dato (1 = el día siguiente)", examples=[1])


def fecha_en_texto(f: date) -> str:
    return f"{f.day} de {MESES[f.month - 1]} de {f.year}"


def dolares(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", " ")


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    artefacto = cargar_artefacto(CARPETA, MODELO_INFO["slug"])
    dias = entrada.dias_adelante
    valor = float(recursiva(artefacto["pipeline"], [artefacto["ultimos_cierres"]], dias)[0, -1])
    amplitud = np.exp(Z95 * artefacto["sigma"] * np.sqrt(dias))
    ultimo = date.fromisoformat(artefacto["fecha_max"])
    destino = ultimo + timedelta(days=dias)
    valor = round(valor, 2)

    texto = (f"El modelo estima que el Bitcoin cerraría alrededor de {dolares(valor)} dólares el {fecha_en_texto(destino)}, "
             f"con un rango del 95 % entre {dolares(valor / amplitud)} y {dolares(valor * amplitud)} dólares. "
             f"Atención: los datos del modelo llegan hasta el {fecha_en_texto(ultimo)} y no conoce el precio actual; "
             "en las pruebas no mejoró de forma demostrable a repetir el último precio conocido, así que es una referencia y no una recomendación de inversión.")
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=valor, unidad=MODELO_INFO["unidad"], texto=texto)

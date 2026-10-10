"""Modelo 10 · Predicción del precio de acciones del S&P 500.

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
Todos los campos tienen valor por defecto: `{}` es una solicitud válida (Apple, un día; así se ejecuta solo con la voz).
`ALIAS_SIMBOLOS` es para el asistente de voz: nombre dicho (sin tilde, en minúscula) -> símbolo.
El dataset termina el 7-feb-2018: el modelo no conoce el precio actual y la respuesta lo dice. Las fechas se cuentan en días hábiles
(lunes a viernes; no se descuentan feriados de la bolsa).
"""
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, cargar_artefacto
from features.modelo_10_sp500.serie import recursiva

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "sp500",
    "nombre": "Predicción del precio de acciones del S&P 500",
    "tipo": "regresion",
    "unidad": "USD",
    "comandos": ["precio de la accion", "bolsa", "sp500"],
}

router = APIRouter(prefix="/api/modelos/sp500", tags=["Modelo 10 · S&P 500"])

SIMBOLOS = ("AAPL", "MSFT", "AMZN", "GOOGL")
NOMBRES = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Google"}
ALIAS_SIMBOLOS = {"apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL"}
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre")
MAX_DIAS = 7
Z95 = 1.96


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos ("3", 2.5, true).
    model_config = ConfigDict(extra="forbid", strict=True)

    simbolo: Literal[SIMBOLOS] = Field("AAPL", description="Símbolo de la acción: AAPL (Apple), MSFT (Microsoft), AMZN (Amazon) o GOOGL (Google)")
    dias_adelante: int = Field(1, ge=1, le=MAX_DIAS, description="Sesiones de bolsa después del último dato (1 = la siguiente)", examples=[1])


def fecha_en_texto(f: date) -> str:
    return f"{f.day} de {MESES[f.month - 1]} de {f.year}"


def dolares(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", " ")


def fecha_destino(ultimo: date, dias: int) -> date:
    """Día hábil número `dias` después de `ultimo` (lunes a viernes)."""
    return date.fromisoformat(str(np.busday_offset(np.datetime64(ultimo), dias, roll="backward")))


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    artefacto = cargar_artefacto(CARPETA, MODELO_INFO["slug"])
    dias, simbolo = entrada.dias_adelante, entrada.simbolo
    valor = float(recursiva(artefacto["pipeline"], [artefacto["ultimos_cierres"][simbolo]], dias)[0, -1])
    amplitud = np.exp(Z95 * artefacto["sigma"][simbolo] * np.sqrt(dias))
    ultimo = date.fromisoformat(artefacto["fecha_max"])
    destino = fecha_destino(ultimo, dias)
    valor = round(valor, 2)

    texto = (f"El modelo estima que la acción de {NOMBRES[simbolo]} ({simbolo}) cerraría alrededor de {dolares(valor)} dólares el {fecha_en_texto(destino)}, "
             f"con un rango del 95 % entre {dolares(valor / amplitud)} y {dolares(valor * amplitud)} dólares. "
             f"Atención: los datos del modelo llegan hasta el {fecha_en_texto(ultimo)} y no conoce el precio actual; "
             "en las pruebas su ventaja sobre repetir el último precio vino solo del crecimiento medio histórico y no de las variables que usa, "
             "así que es una referencia y no una recomendación de inversión.")
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=valor, unidad=MODELO_INFO["unidad"], texto=texto)

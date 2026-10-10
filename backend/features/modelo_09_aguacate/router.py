"""Modelo 09 · Predicción del precio del aguacate.

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
Todos los campos tienen valor por defecto: `{}` es una solicitud válida (así se ejecuta solo con la voz).
"""
import re
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.modelos import RespuestaPrediccion, cargar_artefacto, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "aguacate",
    "nombre": "Predicción del precio del aguacate",
    "tipo": "regresion",
    "unidad": "USD por aguacate",
    "comandos": ["precio del aguacate", "precio de aguacate", "cuanto cuesta el aguacate"],
}

router = APIRouter(prefix="/api/modelos/aguacate", tags=["Modelo 09 · Aguacate"])

# Regiones del dataset (54, incluye agregados como TotalUS y West). Una prueba verifica que coincidan.
REGIONES = (
    "Albany", "Atlanta", "BaltimoreWashington", "Boise", "Boston", "BuffaloRochester", "California", "Charlotte",
    "Chicago", "CincinnatiDayton", "Columbus", "DallasFtWorth", "Denver", "Detroit", "GrandRapids", "GreatLakes",
    "HarrisburgScranton", "HartfordSpringfield", "Houston", "Indianapolis", "Jacksonville", "LasVegas", "LosAngeles",
    "Louisville", "MiamiFtLauderdale", "Midsouth", "Nashville", "NewOrleansMobile", "NewYork", "Northeast",
    "NorthernNewEngland", "Orlando", "Philadelphia", "PhoenixTucson", "Pittsburgh", "Plains", "Portland",
    "RaleighGreensboro", "RichmondNorfolk", "Roanoke", "Sacramento", "SanDiego", "SanFrancisco", "Seattle",
    "SouthCarolina", "SouthCentral", "Southeast", "Spokane", "StLouis", "Syracuse", "Tampa", "TotalUS", "West",
    "WestTexNewMexico",
)
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre")
DIAS_DE_GRACIA = 60  # pasada esta holgura desde el último dato, se avisa que es una estimación de largo plazo


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos (una fecha como número, "frito" como tipo).
    model_config = ConfigDict(extra="forbid", strict=True)

    region: Literal[REGIONES] = Field("TotalUS", description="Región de EE. UU. (TotalUS = todo el país)")
    tipo: Literal["conventional", "organic"] = Field("conventional", description="Convencional u orgánico")
    fecha: date = Field(default_factory=date.today, ge=date(2015, 1, 1), le=date(2100, 12, 31), strict=False,
                        description="Fecha de la estimación, AAAA-MM-DD (por defecto, hoy)")

    @field_validator("fecha", mode="before")
    @classmethod
    def solo_fecha_iso(cls, valor):
        """Con `strict=True` pydantic rechazaría hasta "2017-09-15"; con el modo laxo aceptaría un entero (marca de tiempo) o una
        hora. Se admite únicamente una cadena AAAA-MM-DD (o un objeto `date`, para uso interno)."""
        if isinstance(valor, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", valor):
            raise ValueError("la fecha debe tener el formato AAAA-MM-DD")
        if not isinstance(valor, (str, date)):
            raise ValueError("la fecha debe ser una cadena AAAA-MM-DD")
        return valor


def nombre_region(region: str) -> str:
    """'LosAngeles' -> 'Los Angeles'; 'TotalUS' -> 'todo Estados Unidos'."""
    if region == "TotalUS":
        return "todo Estados Unidos"
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", region).replace("Ft ", "Ft. ").replace("St ", "St. ")


def fecha_en_texto(f: date) -> str:
    return f"{f.day} de {MESES[f.month - 1]} de {f.year}"


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = {"region": entrada.region, "tipo": entrada.tipo, "fecha": pd.Timestamp(entrada.fecha)}
    valor, _ = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    valor = round(max(valor, 0.0), 2)
    tipo = "convencional" if entrada.tipo == "conventional" else "orgánico"
    texto = (f"El precio estimado del aguacate {tipo} en {nombre_region(entrada.region)} "
             f"para el {fecha_en_texto(entrada.fecha)} es de {valor} dólares por unidad.")

    fecha_max = date.fromisoformat(cargar_artefacto(CARPETA, MODELO_INFO["slug"])["fecha_max"])
    if (entrada.fecha - fecha_max).days > DIAS_DE_GRACIA:
        texto += (f" Atención: los datos del modelo llegan hasta el {fecha_en_texto(fecha_max)}; para esta fecha es "
                  "una estimación basada en la estacionalidad y en el último nivel de precios conocido, y es poco confiable.")
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=valor, unidad=MODELO_INFO["unidad"], texto=texto)

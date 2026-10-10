"""Modelo 06 · Clasificación del estado hepático por hepatitis C (donante, hepatitis, fibrosis o cirrosis).

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
NO es un diagnóstico. Las clases raras se ponderaron al entrenar: los puntajes no son probabilidades calibradas.
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "hepatitis",
    "nombre": "Clasificación del estado hepático por hepatitis C",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["tipo de hepatitis", "diagnostico de hepatitis", "estado del higado"],
}

router = APIRouter(prefix="/api/modelos/hepatitis", tags=["Modelo 06 · Hepatitis"])

ORDEN = ("donante", "hepatitis", "fibrosis", "cirrosis")
AVISO = ("Es un ejercicio educativo con un conjunto de datos pequeño; no es un diagnóstico y no reemplaza la valoración "
         "de un profesional de la salud.")


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos (p. ej. `alp`, que el modelo no usa) y tipos laxos ("45", true).
    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=18, le=100, description="Edad en años", examples=[47])
    sex: Literal["f", "m"] = Field(description="Sexo: f (femenino) o m (masculino)")
    alb: float = Field(ge=10, le=90, description="Albúmina (g/L)", examples=[42.0])
    alt: float = Field(ge=0.5, le=400, description="Alanina aminotransferasa, ALT (U/L)", examples=[23.0])
    ast: float = Field(ge=5, le=400, description="Aspartato aminotransferasa, AST (U/L)", examples=[25.0])
    bil: float = Field(ge=1, le=300, description="Bilirrubina (µmol/L)", examples=[7.3])
    che: float = Field(ge=1, le=20, description="Colinesterasa (kU/L)", examples=[8.3])
    chol: float = Field(ge=1, le=12, description="Colesterol (mmol/L)", examples=[5.3])
    crea: float = Field(ge=5, le=1200, description="Creatinina (µmol/L)", examples=[77.0])
    ggt: float = Field(ge=2, le=800, description="Gamma-glutamil transferasa, GGT (U/L)", examples=[23.0])
    prot: float = Field(ge=40, le=100, description="Proteínas totales (g/L)", examples=[72.0])


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    clase, puntajes = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    detalle = ", ".join(f"{c} {round(puntajes[c] * 100)} %" for c in ORDEN)
    sin_enfermedad = " (sin signos de enfermedad hepática en estos datos)" if clase == "donante" else ""
    texto = (f"Según los valores de laboratorio ingresados, el modelo clasifica este perfil como {clase}{sin_enfermedad}. "
             f"Puntajes: {detalle}. Los puntajes no son probabilidades calibradas. {AVISO}")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: algún valor está fuera del rango de las personas con que se entrenó el modelo y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=clase, probabilidades=puntajes, texto=texto)

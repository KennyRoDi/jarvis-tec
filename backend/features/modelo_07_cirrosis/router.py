"""Modelo 07 · Etapa histológica de la cirrosis biliar primaria (1 a 4).

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
NO es un diagnóstico: la etapa se determina por biopsia. Las etapas se ponderaron al entrenar: los puntajes no son
probabilidades calibradas.
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "cirrosis",
    "nombre": "Clasificación de la etapa de cirrosis",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["etapa de cirrosis", "tipo de cirrosis", "estadio de la cirrosis"],
}

router = APIRouter(prefix="/api/modelos/cirrosis", tags=["Modelo 07 · Cirrosis"])

ETAPAS = (1, 2, 3, 4)
AVISO = ("Es un ejercicio educativo con muy pocos pacientes; no es un diagnóstico: la etapa real se determina por biopsia y "
         "esta estimación puede equivocarse por una etapa o más.")
SiNo = Literal["Y", "N"]


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos (p. ej. `n_days` o `status`, que son seguimiento posterior) y tipos laxos.
    model_config = ConfigDict(extra="forbid", strict=True)

    age: float = Field(ge=18, le=100, description="Edad en años", examples=[50.0])
    sex: Literal["F", "M"] = Field(description="Sexo: F (femenino) o M (masculino)")
    ascites: SiNo = Field(description="Ascitis (Y/N)")
    hepatomegaly: SiNo = Field(description="Hepatomegalia (Y/N)")
    spiders: SiNo = Field(description="Angiomas en araña (Y/N)")
    edema: Literal["N", "S", "Y"] = Field(description="Edema: N (no), S (sin diuréticos o resuelto con ellos), Y (a pesar de diuréticos)")
    bilirubin: float = Field(ge=0.1, le=40, description="Bilirrubina sérica (mg/dL)", examples=[1.4])
    cholesterol: float = Field(ge=100, le=2200, description="Colesterol sérico (mg/dL)", examples=[310.0])
    albumin: float = Field(ge=1.5, le=5.5, description="Albúmina (g/dL)", examples=[3.5])
    copper: float = Field(ge=3, le=800, description="Cobre en orina (µg/día)", examples=[74.0])
    alk_phos: float = Field(ge=200, le=18000, description="Fosfatasa alcalina (U/L)", examples=[1277.0])
    sgot: float = Field(ge=20, le=600, description="SGOT/AST (U/mL)", examples=[116.0])
    tryglicerides: float = Field(ge=25, le=800, description="Triglicéridos (mg/dL)", examples=[108.0])
    platelets: float = Field(ge=40, le=700, description="Plaquetas (miles por mL)", examples=[257.0])
    prothrombin: float = Field(ge=7, le=22, description="Tiempo de protrombina (segundos)", examples=[10.6])


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    etapa, puntajes = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    detalle = ", ".join(f"etapa {e} {round(puntajes[str(e)] * 100)} %" for e in ETAPAS)
    texto = (f"Según los datos ingresados, el modelo estima la etapa histológica {etapa} de 4 (1 es la más temprana y 4 es cirrosis). "
             f"Puntajes: {detalle}. Los puntajes no son probabilidades calibradas. {AVISO}")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: algún valor está fuera del rango de los pacientes con que se entrenó el modelo y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=int(etapa), probabilidades=puntajes, texto=texto)

"""Modelo 05 · Riesgo de accidente cerebrovascular (ACV).

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
`prediccion` es "Yes" si la probabilidad alcanza el umbral F1 del artefacto (riesgo alto). El texto añade un nivel
intermedio ("moderado") con el umbral de sensibilidad. NO es un diagnóstico: es una estimación estadística.
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, cargar_artefacto, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "acv",
    "nombre": "Clasificación del riesgo de accidente cerebrovascular",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["riesgo de derrame", "accidente cerebrovascular", "riesgo de acv"],
}

router = APIRouter(prefix="/api/modelos/acv", tags=["Modelo 05 · ACV"])

AVISO = "Es una estimación estadística con fines educativos y no reemplaza la valoración de un profesional de la salud."


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos ("45", true).
    model_config = ConfigDict(extra="forbid", strict=True)

    age: float = Field(ge=0, le=120, description="Edad en años", examples=[67.0])
    hypertension: Literal["Yes", "No"] = Field(description="Hipertensión diagnosticada")
    heart_disease: Literal["Yes", "No"] = Field(description="Enfermedad cardíaca diagnosticada")
    avg_glucose_level: float = Field(ge=40, le=400, description="Nivel promedio de glucosa en sangre (mg/dL)", examples=[105.0])
    bmi: float = Field(ge=10, le=100, description="Índice de masa corporal (kg/m²)", examples=[28.5])


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    _, probabilidades = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    artefacto = cargar_artefacto(CARPETA, MODELO_INFO["slug"])
    p = probabilidades["Yes"]
    clase = "Yes" if p >= artefacto["umbral"] else "No"
    nivel = "alto" if p >= artefacto["umbral"] else "moderado" if p >= artefacto["umbral_sensibilidad"] else "bajo"
    porcentaje = round(p * 100, 1)
    texto = (f"Según los datos ingresados, la probabilidad estimada de sufrir un accidente cerebrovascular es de "
             f"{porcentaje} por ciento, lo que corresponde a un riesgo {nivel}. {AVISO}")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: alguna medida está fuera del rango de los pacientes con que se entrenó el modelo y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=clase, probabilidades=probabilidades, texto=texto)

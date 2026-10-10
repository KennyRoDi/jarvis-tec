"""Modelo 08 · Predicción del porcentaje de grasa corporal.

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
Los límites de Entrada son los del dataset completo con un margen (rechazan datos absurdos); el aviso de
extrapolación usa el rango de entrenamiento, que es más estrecho (ver analisis.md).
"""
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from core.modelos import RespuestaPrediccion, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "grasa_corporal",
    "nombre": "Predicción del porcentaje de grasa corporal",
    "tipo": "regresion",
    "unidad": "%",
    "comandos": ["grasa corporal", "masa corporal", "porcentaje de grasa"],
}

router = APIRouter(prefix="/api/modelos/grasa_corporal", tags=["Modelo 08 · Grasa corporal"])


class Entrada(BaseModel):
    # Estricta: rechaza campos desconocidos y tipos laxos ("45", true, 45.5 en la edad).
    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=18, le=90, description="Edad en años", examples=[45])
    weight_kg: float = Field(ge=45, le=180, description="Peso en kg", examples=[80.0])
    height_cm: float = Field(ge=145, le=205, description="Estatura en cm", examples=[178.0])
    neck_cm: float = Field(ge=28, le=55, description="Circunferencia del cuello (cm)", examples=[38.0])
    chest_cm: float = Field(ge=75, le=145, description="Circunferencia del pecho (cm)", examples=[100.0])
    abdomen_cm: float = Field(ge=65, le=155, description="Circunferencia del abdomen (cm)", examples=[92.0])
    hip_cm: float = Field(ge=80, le=150, description="Circunferencia de la cadera (cm)", examples=[99.0])
    thigh_cm: float = Field(ge=43, le=90, description="Circunferencia del muslo (cm)", examples=[59.0])
    knee_cm: float = Field(ge=30, le=52, description="Circunferencia de la rodilla (cm)", examples=[38.5])
    ankle_cm: float = Field(ge=17, le=36, description="Circunferencia del tobillo (cm)", examples=[23.0])
    biceps_cm: float = Field(ge=22, le=48, description="Circunferencia del bíceps (cm)", examples=[32.0])
    forearm_cm: float = Field(ge=19, le=37, description="Circunferencia del antebrazo (cm)", examples=[28.7])
    wrist_cm: float = Field(ge=14, le=24, description="Circunferencia de la muñeca (cm)", examples=[18.3])


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    valor, _ = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    valor = round(max(valor, 0.0), 1)  # un porcentaje no puede ser negativo
    texto = (f"Con esas medidas, el porcentaje de grasa corporal estimado es {valor} por ciento. "
             "Es una estimación orientativa, calculada con datos de hombres adultos.")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: alguna medida está fuera del rango de los datos de entrenamiento y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=valor, unidad=MODELO_INFO["unidad"], texto=texto)

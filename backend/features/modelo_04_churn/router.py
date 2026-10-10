"""Modelo 04 · Clasificación de abandono de clientes de telefonía (churn).

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
`prediccion` es "Yes" (abandona) o "No" según el umbral guardado en el artefacto (no el 0.5 por defecto).
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from core.modelos import RespuestaPrediccion, cargar_artefacto, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "churn",
    "nombre": "Clasificación de abandono de clientes de telefonía",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["cliente se va", "abandono de cliente", "churn", "se va a pasar de compania"],
}

router = APIRouter(prefix="/api/modelos/churn", tags=["Modelo 04 · Churn"])

SiNo = Literal["Yes", "No"]
SiNoSinInternet = Literal["Yes", "No", "No internet service"]


class Entrada(BaseModel):
    tenure: int = Field(ge=0, le=120, description="Meses como cliente", examples=[12])
    monthly_charges: float = Field(ge=15, le=130, description="Mensualidad en USD", examples=[70.0])
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(description="Tipo de contrato")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(description="Servicio de internet")
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] = Field(
        description="Método de pago")
    paperless_billing: SiNo = Field(description="Factura electrónica")
    tech_support: SiNoSinInternet = Field(description="Soporte técnico (si no hay internet: 'No internet service')")
    online_security: SiNoSinInternet = Field(description="Seguridad en línea (si no hay internet: 'No internet service')")
    senior_citizen: SiNo = Field(description="Adulto mayor")

    @model_validator(mode="after")
    def servicios_de_internet_coherentes(self):
        """Sin internet los servicios adicionales no existen, y con internet siempre son 'Yes' o 'No' (así es en los datos)."""
        sin_internet = self.internet_service == "No"
        for campo in ("tech_support", "online_security"):
            if (getattr(self, campo) == "No internet service") != sin_internet:
                raise ValueError(f"{campo} debe ser 'No internet service' si y solo si internet_service es 'No'")
        return self


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    _, probabilidades = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    umbral = cargar_artefacto(CARPETA, MODELO_INFO["slug"])["umbral"]
    p_abandona = probabilidades["Yes"]
    clase = "Yes" if p_abandona >= umbral else "No"
    riesgo = "alto" if clase == "Yes" else "bajo"
    texto = (f"Este cliente tiene una probabilidad del {round(p_abandona * 100)} por ciento de abandonar la compañía, "
             f"por lo que se clasifica como de riesgo {riesgo}. El modelo marca riesgo alto desde el {round(umbral * 100)} por ciento.")
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: alguna medida está fuera del rango de los clientes con que se entrenó el modelo y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=clase, probabilidades=probabilidades, texto=texto)

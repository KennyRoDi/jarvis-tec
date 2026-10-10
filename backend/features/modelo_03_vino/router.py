"""Modelo 03 · Clasificación de la calidad del vino (baja, media o alta).

Contrato: specs/api_rest_spec.md §3 · Entrada y criterios: SPEC.md de esta carpeta.
Los límites de Entrada son los del dataset con un margen: rechazan datos absurdos (p. ej. un pH de 14).
"""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from core.modelos import RespuestaPrediccion, fuera_de_rango, predecir_con_pipeline

CARPETA = Path(__file__).parent

MODELO_INFO = {
    "slug": "vino",
    "nombre": "Clasificación de la calidad del vino",
    "tipo": "clasificacion",
    "unidad": None,
    "comandos": ["calidad del vino", "clasificar vino", "que tan bueno es el vino"],
}

router = APIRouter(prefix="/api/modelos/vino", tags=["Modelo 03 · Vino"])

ORDEN = ("baja", "media", "alta")
UMBRAL_POCO_CONCLUYENTE = 0.5  # si la clase más probable no llega a esto, se avisa


class Entrada(BaseModel):
    tipo: Literal["red", "white"] = Field(description="Tinto (red) o blanco (white)", examples=["white"])
    fixed_acidity: float = Field(ge=3, le=17, description="Acidez fija (g/dm³, ácido tartárico)", examples=[7.0])
    volatile_acidity: float = Field(ge=0.05, le=2, description="Acidez volátil (g/dm³, ácido acético)", examples=[0.3])
    citric_acid: float = Field(ge=0, le=2, description="Ácido cítrico (g/dm³)", examples=[0.32])
    residual_sugar: float = Field(ge=0.3, le=70, description="Azúcar residual (g/dm³)", examples=[2.0])
    chlorides: float = Field(ge=0.005, le=0.7, description="Cloruros (g/dm³, cloruro de sodio)", examples=[0.045])
    free_sulfur_dioxide: float = Field(ge=0.5, le=320, description="Dióxido de azufre libre (mg/dm³)", examples=[30.0])
    total_sulfur_dioxide: float = Field(ge=5, le=460, description="Dióxido de azufre total (mg/dm³)", examples=[115.0])
    density: float = Field(ge=0.98, le=1.05, description="Densidad (g/cm³)", examples=[0.994])
    ph: float = Field(ge=2.6, le=4.2, description="pH", examples=[3.2])
    sulphates: float = Field(ge=0.2, le=2.1, description="Sulfatos (g/dm³, sulfato de potasio)", examples=[0.5])
    alcohol: float = Field(ge=7.5, le=15.5, description="Grado alcohólico (% vol.)", examples=[10.5])

    @model_validator(mode="after")
    def azufre_libre_no_supera_al_total(self):
        """El dióxido de azufre libre es una parte del total (en el dataset nunca lo supera)."""
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError("free_sulfur_dioxide no puede ser mayor que total_sulfur_dioxide")
        return self


@router.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: Entrada) -> RespuestaPrediccion:
    datos = entrada.model_dump()
    clase, probabilidades = predecir_con_pipeline(CARPETA, MODELO_INFO["slug"], datos)
    porcentajes = ", ".join(f"{c} {round(probabilidades[c] * 100)} %" for c in ORDEN)
    tipo = "tinto" if entrada.tipo == "red" else "blanco"
    texto = (f"Según sus propiedades fisicoquímicas, este vino {tipo} se clasifica como de calidad {clase}, "
             f"con una probabilidad del {round(probabilidades[clase] * 100)} por ciento. Probabilidades: {porcentajes}.")
    if probabilidades[clase] < UMBRAL_POCO_CONCLUYENTE:
        texto += " La clasificación es poco concluyente: ninguna clase supera el 50 por ciento."
    if fuera_de_rango(CARPETA, MODELO_INFO["slug"], datos):
        texto += " Atención: alguna medida está fuera del rango de los vinos con que se entrenó el modelo y el resultado es poco confiable."
    return RespuestaPrediccion(modelo=MODELO_INFO["slug"], prediccion=clase, probabilidades=probabilidades, texto=texto)

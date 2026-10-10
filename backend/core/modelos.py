"""Piezas comunes a los 10 modelos de ML: registro, carga de artefactos y predicción.

Cada `features/modelo_XX_<slug>/router.py` exporta `MODELO_INFO` y `router`; `main.py` los registra aquí.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from core.errores import ApiError

ARTEFACTO = "modelo.joblib"
METRICAS = "metricas.json"


@dataclass
class ModeloRegistrado:
    id: str
    carpeta: Path
    info: dict
    entrada: type[BaseModel] | None = None  # esquema `Entrada` del router.py

    @property
    def entrenado(self) -> bool:
        return (self.carpeta / ARTEFACTO).exists()

    def resumen(self) -> dict:
        campos = ("slug", "nombre", "tipo", "comandos")
        return {"id": self.id, **{c: self.info.get(c) for c in campos}, "entrenado": self.entrenado}


REGISTRO: dict[str, ModeloRegistrado] = {}  # slug -> modelo


def registrar(id_feature: str, carpeta: Path, info: dict, entrada: type[BaseModel] | None = None) -> None:
    REGISTRO[info["slug"]] = ModeloRegistrado(id=id_feature, carpeta=carpeta, info=info, entrada=entrada)


class RespuestaPrediccion(BaseModel):
    modelo: str
    prediccion: float | int | str | list
    unidad: str | None = None
    probabilidades: dict[str, float] | None = None
    texto: str


# Caché por (ruta, mtime): si se reentrena con el servidor arriba, se recarga el artefacto nuevo.
_cache: dict[Path, tuple[float, dict]] = {}


def cargar_artefacto(carpeta: Path, slug: str) -> dict:
    ruta = carpeta / ARTEFACTO
    if not ruta.exists():
        raise ApiError(503, "MODELO_NO_ENTRENADO", f"El modelo '{slug}' aún no ha sido entrenado.")
    mtime = ruta.stat().st_mtime
    if ruta not in _cache or _cache[ruta][0] != mtime:
        _cache[ruta] = (mtime, joblib.load(ruta))
    return _cache[ruta][1]


def _a_python(valor):
    return valor.item() if isinstance(valor, np.generic) else valor


def predecir_con_pipeline(carpeta: Path, slug: str, entrada: dict) -> tuple[object, dict | None]:
    """Ejecuta el pipeline de sklearn guardado sobre una sola fila. Devuelve (predicción, probabilidades)."""
    pipeline = cargar_artefacto(carpeta, slug)["pipeline"]
    fila = pd.DataFrame([entrada])
    prediccion = _a_python(pipeline.predict(fila)[0])
    probabilidades = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(fila)[0]
        probabilidades = {str(c): round(float(p), 4) for c, p in zip(pipeline.classes_, probs)}
    return prediccion, probabilidades


def fuera_de_rango(carpeta: Path, slug: str, entrada: dict, margen: float = 0.05) -> list[str]:
    """Campos de `entrada` fuera del rango visto en el entrenamiento (más un `margen` del ancho del rango).

    Requiere que el artefacto guarde `rango={campo: [mínimo, máximo]}` (ver `guardar_modelo(..., rango=...)`).
    Los modelos extrapolan mal fuera de lo que vieron: el router usa esto para avisar que el resultado es poco confiable.
    """
    rango = cargar_artefacto(carpeta, slug).get("rango", {})
    return [campo for campo, (minimo, maximo) in rango.items()
            if not minimo - (maximo - minimo) * margen <= entrada[campo] <= maximo + (maximo - minimo) * margen]


def leer_metricas(carpeta: Path) -> dict | None:
    ruta = carpeta / METRICAS
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


router_modelos = APIRouter(prefix="/api/modelos", tags=["Modelos"])


@router_modelos.get("")
def listar_modelos():
    return {"modelos": [m.resumen() for m in sorted(REGISTRO.values(), key=lambda m: m.id)]}


@router_modelos.get("/{slug}/info")
def info_modelo(slug: str):
    modelo = REGISTRO.get(slug)
    if modelo is None:
        raise ApiError(404, "NO_ENCONTRADO", f"No existe el modelo '{slug}'.")
    metricas = leer_metricas(modelo.carpeta)
    return {
        **modelo.resumen(),
        "metricas": (metricas or {}).get("metricas") if modelo.entrenado else None,
        "entrada_ejemplo": (metricas or {}).get("entrada_ejemplo"),
        # JSON Schema de la entrada: el frontend genera el formulario de cada modelo a partir de aquí.
        "esquema_entrada": modelo.entrada.model_json_schema() if modelo.entrada else None,
    }

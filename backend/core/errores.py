"""Formato de error uniforme (specs/api_rest_spec.md §1.1)."""
import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("jarvis")

CODIGOS_HTTP = {
    400: "SOLICITUD_INVALIDA",
    404: "NO_ENCONTRADO",
    415: "TIPO_NO_SOPORTADO",
    422: "VALIDACION",
    500: "ERROR_INTERNO",
    501: "NO_IMPLEMENTADO",
    502: "SERVICIO_EXTERNO",
    503: "MODELO_NO_ENTRENADO",
}


class ApiError(Exception):
    """Error de negocio que se traduce directamente a la respuesta JSON del contrato."""

    def __init__(self, status: int, codigo: str, mensaje: str, detalle=None):
        super().__init__(mensaje)
        self.status = status
        self.codigo = codigo
        self.mensaje = mensaje
        self.detalle = detalle


def respuesta_error(status: int, codigo: str, mensaje: str, detalle=None) -> JSONResponse:
    cuerpo = {"error": {"codigo": codigo, "mensaje": mensaje, "detalle": detalle}}
    return JSONResponse(status_code=status, content=jsonable_encoder(cuerpo))


def registrar_manejadores(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return respuesta_error(exc.status, exc.codigo, exc.mensaje, exc.detalle)

    @app.exception_handler(RequestValidationError)
    async def _validacion(_: Request, exc: RequestValidationError):
        # Solo loc/msg/type: `input` puede traer NaN (JSON no estándar) y `ctx` objetos no serializables.
        detalle = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
        return respuesta_error(422, "VALIDACION", "La solicitud no cumple el esquema.", detalle)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        codigo = CODIGOS_HTTP.get(exc.status_code, "SOLICITUD_INVALIDA" if exc.status_code < 500 else "ERROR_INTERNO")
        return respuesta_error(exc.status_code, codigo, str(exc.detail))

    @app.exception_handler(Exception)
    async def _no_controlado(_: Request, exc: Exception):
        logger.exception("Error no controlado", exc_info=exc)
        return respuesta_error(500, "ERROR_INTERNO", "Ocurrió un error interno en el servidor.")

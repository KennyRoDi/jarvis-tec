"""Punto de entrada de JarvisTEC: API FastAPI + ventana de escritorio PyWebView.

Los endpoints de cada feature se descubren solos: cualquier `features/<carpeta>/router.py` que exporte
`router` se incluye, y si además exporta `MODELO_INFO` se registra como modelo de ML.
NO se edita este archivo para agregar rutas (regla ADD, ver CLAUDE.md).

App de escritorio:  python backend/main.py
Solo API:           cd backend && uvicorn main:app --reload
"""
import importlib
import logging
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core import modelos
from core.config import CORS_ORIGENES, FEATURES_DIR, HOST, PUERTO, UI_PRUEBA_DIR, UI_REACT_DIR, VERSION
from core.errores import registrar_manejadores
from core.escritorio import permitir_camara_y_microfono

logger = logging.getLogger("jarvis")

app = FastAPI(title="JarvisTEC API", version=VERSION, description="Contrato: specs/api_rest_spec.md")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGENES, allow_methods=["*"], allow_headers=["*"])
registrar_manejadores(app)


@app.get("/api/salud", tags=["Sistema"])
def salud():
    return {"estado": "ok", "version": VERSION}


def cargar_features() -> None:
    for carpeta in sorted(p for p in FEATURES_DIR.iterdir() if (p / "router.py").exists()):
        try:
            modulo = importlib.import_module(f"features.{carpeta.name}.router")
        except Exception:
            # Una feature rota no debe tumbar la API del resto del equipo.
            logger.exception("No se pudo cargar la feature '%s'", carpeta.name)
            continue
        if hasattr(modulo, "MODELO_INFO"):
            modelos.registrar(carpeta.name, carpeta, modulo.MODELO_INFO, getattr(modulo, "Entrada", None))
        app.include_router(modulo.router)


# /api/modelos y /api/modelos/{slug}/info van antes que las rutas de cada modelo.
app.include_router(modelos.router_modelos)
cargar_features()

# La interfaz se monta al final para que nunca tape una ruta /api.
UI_DIR = UI_REACT_DIR if (UI_REACT_DIR / "index.html").exists() else UI_PRUEBA_DIR
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


def iniciar_servidor() -> uvicorn.Server:
    """Arranca uvicorn en un hilo daemon (muere al cerrar la ventana) y espera a que esté listo."""
    servidor = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PUERTO, log_level="info"))
    hilo = threading.Thread(target=servidor.run, daemon=True, name="fastapi")
    hilo.start()
    limite = time.monotonic() + 10
    while not servidor.started:
        if not hilo.is_alive() or time.monotonic() > limite:
            raise RuntimeError(f"El servidor no arrancó en {HOST}:{PUERTO} (¿puerto ocupado?)")
        time.sleep(0.05)
    return servidor


def main() -> None:
    import webview  # solo la app de escritorio lo necesita; uvicorn y las pruebas no

    servidor = iniciar_servidor()
    ventana = webview.create_window(
        "JarvisTEC", f"http://{HOST}:{PUERTO}/", width=1280, height=800, min_size=(960, 600)
    )
    webview.start(permitir_camara_y_microfono, ventana)
    servidor.should_exit = True


if __name__ == "__main__":
    main()

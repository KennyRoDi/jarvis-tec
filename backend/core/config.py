"""Configuración leída de backend/.env (ver .env.example y specs/api_rest_spec.md §6)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BACKEND_DIR / "features"

load_dotenv(BACKEND_DIR / ".env")

VERSION = "0.1.0"
HOST = os.getenv("JARVIS_HOST", "127.0.0.1")
PUERTO = int(os.getenv("JARVIS_PUERTO", "8000"))

# La ventana de escritorio sirve el build de React si existe; si no, la página de prueba.
UI_REACT_DIR = BACKEND_DIR.parent / "frontend" / "dist"
UI_PRUEBA_DIR = BACKEND_DIR / "ui_prueba"

CORS_ORIGENES = [o.strip() for o in os.getenv("CORS_ORIGENES", "http://localhost:5173").split(",") if o.strip()]

AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT", "")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY", "")

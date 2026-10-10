# JarvisTEC-Core — Memoria del proyecto (ADD)

Asistente personal de escritorio estilo "Jarvis" (Proyecto IA, I Semestre 2026, TEC San Carlos).
Este archivo es la memoria persistente de los agentes: si cambias una convención, comando o el estado
del proyecto, actualízalo aquí en el mismo commit.

## Arquitectura

```
PyWebView (ventana nativa)  ──►  http://127.0.0.1:8000/       interfaz web (frontend/dist o backend/ui_prueba)
                                 http://127.0.0.1:8000/api/*  FastAPI (hilo en segundo plano)
                                        └─ backend/features/<feature>/router.py  (registro automático)
```

- **SDD — contrato primero.** `specs/api_rest_spec.md` es la única fuente de verdad. Antes de codificar o
  cambiar un endpoint, se actualiza la spec (y su historial). El backend la implementa; el frontend la
  consume y la mockea en `frontend/src/api/mocks.js`.
- **FDD — rebanadas verticales.** Cada modelo de ML o servicio cognitivo es una carpeta aislada en
  `backend/features/` con sus datos, entrenamiento y `router.py`. `specs/modelos_spec.md` lista los 10 modelos.
- **ADD — agentes.** El agente escribe código funcional con pruebas, abstrae lo repetitivo en `backend/core/`
  y mantiene este archivo al día.

## Reglas estrictas

0. **Lee el `SPEC.md` de la carpeta antes de trabajar en ella** (cada feature, `core/`, `frontend/` y
   `docs_latex/` tienen uno), y marca sus criterios de aceptación al terminar. `specs/alcance_spec.md`
   relaciona cada requisito del enunciado con su carpeta: es el checklist de la entrega.

1. **No modificar `backend/main.py` para agregar rutas.** Toda feature expone `router` (un `APIRouter`) en
   `backend/features/<feature>/router.py`; `main.py` lo descubre e incluye solo. Si además exporta
   `MODELO_INFO`, queda registrada como modelo de ML (`GET /api/modelos`).
2. **Escritorio = PyWebView.** No usar Electron, Tauri, Flet, Eel ni similares. La ventana carga la UI
   desde el mismo servidor FastAPI (mismo origen, sin CORS). El frontend usa rutas relativas `/api/...`.
3. **API = FastAPI.** Respuestas JSON limpias. Los errores se lanzan con `core.errores.ApiError(status, codigo, mensaje)`
   y siempre salen con el formato `{"error": {"codigo", "mensaje", "detalle"}}` de la spec (§1.1).
4. **Servicios cloud obligatorios.** Rostros: Azure Face; emoción: Google Cloud Vision (Azure retiró `emotion`). No inventar modelos locales.
   Voz a texto: Google Cloud Speech-to-Text. Credenciales solo en `backend/.env` (ver `backend/.env.example`).
5. **Paralelismo (2 devs).** No tocar archivos fuera de la feature asignada.
   - Dev A: `backend/features/modelo_XX_*`.
   - Dev B: `frontend/`, `backend/features/asistente_voz`, `backend/features/vision_facial`.
   - Cambios a `specs/`, `backend/core/` o `main.py` se acuerdan entre ambos.

## Convenciones de código

- Idioma: español para nombres, campos JSON y mensajes; `snake_case` en Python y JSON.
- Carpeta de modelo: `modelo_XX_<slug>/` con `dataset.csv`, `train.py`, `router.py`, `analisis.md`.
  El entrenamiento genera `modelo.joblib`, `metricas.json` y `figuras/`.
- `train.py` sigue las 6 etapas de la rúbrica: Análisis, Entendimiento, Exploración, Modelo, Evaluación,
  Conclusión. Usa los helpers de `core/entrenamiento.py`. Referencia completa: `modelo_02_autos`.
- `router.py` de modelo: `MODELO_INFO`, esquema `Entrada` (pydantic) y `POST /predecir` usando
  `core.modelos.predecir_con_pipeline`.
- Datos crudos grandes o compartidos en `data/`; cada feature guarda su `dataset.csv`.
- Pruebas: las transversales en `backend/tests/`; las de cada feature en su carpeta (`test_*.py`), usando
  las fixtures `cliente` y `assert_error` de `backend/conftest.py`. Todo endpoint nuevo lleva su prueba.

## Comandos

| Acción                     | Comando                                                           |
|----------------------------|-------------------------------------------------------------------|
| Instalar dependencias      | `pip install -r requirements.txt`                                 |
| App de escritorio          | `python backend/main.py`                                          |
| Solo API (desarrollo)      | `cd backend && uvicorn main:app --reload` → `/docs`               |
| Pruebas                    | `cd backend && pytest`                                            |
| Entrenar un modelo         | `cd backend && python -m features.modelo_XX_<slug>.train`         |
| Frontend (desarrollo)      | `cd frontend && npm run dev` (proxy `/api` → `:8000`)             |
| Frontend con mocks         | `cd frontend && VITE_USAR_MOCKS=true npm run dev`                 |
| Frontend para escritorio   | `cd frontend && npm run build` (PyWebView sirve `frontend/dist`)  |

**Linux:** si se lanza desde la terminal del VS Code instalado por snap, GTK falla por bibliotecas de snap: usar una terminal normal. PyWebView usa GTK/WebKit2 del sistema (`python3-gi`, `gir1.2-webkit2-4.1`); crear el venv con
`python3 -m venv --system-site-packages venv`. **Windows:** usa Edge WebView2, no requiere nada extra.

## Notas operativas

- **Entorno:** Python 3.12+. `kaggle` es solo una herramienta de descarga y no está en `requirements.txt`.
- **Datos:** el `dataset.csv` de cada modelo está versionado; el S&P 500 completo (`data/all_stocks_5yr.csv`)
  no. Se regeneran con `bash data/descargar_datasets.sh` (token de Kaggle en `~/.kaggle/`; en Windows, Git Bash).
  Los SPEC de cada modelo traen trampas verificadas en los datos (p. ej. `Density` en grasa corporal y
  `N_Days`/`Status` en cirrosis son fuga de información): leerlas antes de elegir variables.
- **Un modelo está terminado** solo si cumple las tres cosas de la rúbrica: entrenado y documentado
  (`analisis.md`), endpoint con prueba, y usable desde la interfaz. Ver `specs/alcance_spec.md`.
- **Credenciales:** nunca en el repo ni en el chat. `backend/.env` y el JSON de Google viven en la máquina de
  cada desarrollador. Sin ellas, voz y emoción responden 501; todo lo demás funciona.
- **Cuotas gratuitas:** Azure Face F0 y Google Vision/Speech son limitadas (Vision, del orden de 1000 imágenes
  al mes; verificar la cuota vigente en cada consola). Para no agotarlas: un frame cada 3–5 s como máximo,
  llamar a Vision solo si Azure detectó un rostro, y desarrollar la interfaz con `VITE_USAR_MOCKS=true`.
- **SDK de Azure:** `azure-cognitiveservices-vision-face` está deprecado, pero es el exigido por el equipo y
  la detección de rostros funciona con él. No intentar obtener `emotion` de Azure.
- **Git:** commits con Conventional Commits. `CONTRIBUTING.md` pide ramas `feature/...` y PR; hasta ahora
  se subió directo a `main` solo cuando el usuario lo pidió explícitamente. No subir a `main` por iniciativa propia.
- **Overleaf:** la sincronización con GitHub es de pago; el documento se sube a mano desde `docs_latex/`.
- **Licencias:** los datasets de Kaggle tienen licencias distintas (CC0, ODbL, "copyright-authors"). Revisarlas
  antes de hacer público el repositorio.

## Estado del proyecto

_Actualizar al cerrar cada tarea._ **Entrega: semana 11, tentativa** (puede moverse por un inconveniente aún no definido; no planificar con holgura).

| Componente                   | Estado                                                                 |
|------------------------------|------------------------------------------------------------------------|
| Contrato API                 | v0.1 (`specs/api_rest_spec.md`)                                        |
| Ventana PyWebView + FastAPI  | ✅ `backend/main.py` (UI de prueba en `backend/ui_prueba/`)            |
| Cámara/micrófono en la ventana | ✅ Linux (`core/escritorio.py`, verificado) · ⏳ Windows: lo verifica el compañero |
| Registro automático features | ✅                                                                     |
| Modelo 02 autos              | ✅ entrenado (R² 0.962)                                                |
| Modelos 01, 03–10            | ⏳ plantillas con TODO; `dataset.csv` de los 10 ya está en su carpeta (verificado) |
| Voz a texto (Google)         | ⏳ endpoint valida archivo, responde 501                               |
| Emociones                    | ⏳ responde 501. Decidido y avisado al profesor (2026-10-09): Azure detecta el rostro, Google Vision da la emoción. **Condición del profesor: poder justificarlo en el documento** (`docs_latex/SPEC.md`) |
| Comandos de voz → modelo     | 🟡 reconoce el modelo; faltan parámetros y el tono según la emoción   |
| Interfaz Jarvis (React)      | ⏳ scaffold de Vite + cliente API + mocks                              |
| Documento LaTeX              | ⏳ esqueleto en `docs_latex/main.tex`                                  |

## Documentación científica

Al redactar resultados de modelos: tono académico (tercera persona, justificación formal con referencias),
en Markdown fácil de pasar a LaTeX/Overleaf, dentro del `analisis.md` de cada modelo.

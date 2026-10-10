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
- **Notebook de Colab** (extra, no reemplaza a `train.py`): cada modelo tiene `notebook.ipynb`, autocontenido (no importa del repositorio; en Colab se
  sube solo el `dataset.csv`), con la estructura de `Ejemplo_Plantilla.ipynb` (análisis del problema, hilo conductor, librerías, secciones de
  entendimiento, exploración y modelo, conclusiones), en español, **sin emojis** y con las salidas guardadas. Se genera con
  `herramientas/notebooks/modelo_XX_<slug>.py` (ver `herramientas/notebooks/README.md`); sus cifras deben coincidir con `metricas.json`. Lo vigila
  `backend/tests/test_notebooks.py`.
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
| Regenerar un notebook      | `python herramientas/notebooks/modelo_XX_<slug>.py` (necesita `nbclient` e `ipykernel`) |
| Frontend (desarrollo)      | `cd frontend && npm run dev` (proxy `/api` → `:8000`)             |
| Frontend con mocks         | `cd frontend && VITE_USAR_MOCKS=true npm run dev`                 |
| Frontend para escritorio   | `cd frontend && npm run build` (PyWebView sirve `frontend/dist`)  |

**Linux:** si se lanza desde la terminal del VS Code instalado por snap, GTK falla por bibliotecas de snap: usar una terminal normal. PyWebView usa GTK/WebKit2 del sistema (`python3-gi`, `gir1.2-webkit2-4.1`); crear el venv con
`python3 -m venv --system-site-packages venv`. **Windows:** usa Edge WebView2, no requiere nada extra.

## Orden de desarrollo de los modelos

De menor a mayor complejidad; se trabajan en este orden y cada uno se marca al terminar su `SPEC.md`.

| Orden | Modelo | Complejidad (por qué) | Estado |
|-------|--------|------------------------|--------|
| —  | 02 autos           | Referencia: 301 filas, sin nulos | ✅ |
| 1  | 08 grasa corporal  | 250 filas numéricas, sin nulos; solo excluir `Density` y 2 registros imposibles | ✅ (2026-10-09) |
| 2  | 09 aguacate        | Sin nulos, pero 54 regiones, fechas y valores por defecto para usarlo con la voz | ✅ (2026-10-09) |
| 3  | 03 vino            | Multiclase: agrupar `quality`, imputar 38 nulos, estratificar | ✅ (2026-10-09) |
| 4  | 04 churn           | Binaria, muchas categóricas, `TotalCharges` sucia, entrada de ~20 campos | ⏳ **siguiente** |
| 5  | 05 acv             | 4.9 % de positivos: métricas distintas a accuracy, pesos de clase, umbral | ⏳ |
| 6  | 06 hepatitis       | Multiclase muy desbalanceada, clase de 7 filas | ⏳ |
| 7  | 07 cirrosis        | 418 filas, 106 casi vacías, fuga de información, 4 etapas desiguales | ⏳ |
| 8  | 01 bitcoin         | Serie temporal: fechas, `-`, partición temporal, rezagos, predicción recursiva | ⏳ |
| 9  | 10 sp500           | Serie temporal multi-símbolo; interpretar el símbolo desde la voz | ⏳ |

**Lecciones del modelo 08 que aplican a los demás:**
- Seleccionar el algoritmo solo con validación cruzada sobre el entrenamiento y usar el conjunto de prueba una vez.
- Reportar con honestidad si el R²/F1 de prueba difiere del de validación cruzada, e investigar la causa
  (en el 08, un único registro extremo) en vez de ajustar hasta que el número guste.
- Guardar el rango de entrenamiento en el artefacto (`rango=`) y avisar en `texto` cuando se extrapola.
- El formulario usa unidades del usuario (kg, cm): convertir en `train.py`, no en el router.
- `entrada_ejemplo`: usar `X_test.iloc[[0]].to_dict("records")[0]` (conserva los enteros); con `.iloc[0]` pandas
  convierte todo a `float` y el formulario recibiría `23.0` en un campo entero.
- Las pruebas de límites deben ejercitar de verdad la línea que protegen (p. ej. afirmar que la predicción
  cruda es negativa antes de comprobar el recorte a 0); un test que pasa sin la línea es un test débil.
- Las referencias de `analisis.md` se verifican por búsqueda web (autores, páginas, DOI) antes de pasarlas a
  `docs_latex/referencias.bib`; solo entran al `.bib` las verificadas.
- Revisar la literatura del dataset: suele documentar registros erróneos conocidos (en el 08, los casos 42, 48,
  76, 96 y 182) y conviene citarlos al justificar la limpieza.
- Series de tiempo: partición temporal (no aleatoria), validación cruzada de ventana creciente con las filas de una
  misma fecha juntas, y la selección solo con el entrenamiento. Evitar mezclar `mes` con la semana ISO (se
  contradicen en año nuevo); si un transformador propio vive en el `Pipeline`, debe estar en su propio módulo.
- `HistGradientBoostingRegressor`: fijar `early_stopping=False` (con `auto` depende del tamaño de la muestra).
- **Revisar siempre `df.duplicated()` antes de dividir**: en el vino, 18 % de filas repetidas inflaban la exactitud
  ~10 puntos. Demostrarlo con un experimento en `metricas.json` en vez de solo afirmarlo.
- Clasificación: partición y validación **estratificadas**, F1 macro como criterio si hay desbalance, matriz de
  confusión con las clases en su orden natural y `texto` que avise cuando ninguna clase supera el 50 %.
- **Tamaño del artefacto**: el `.joblib` se versiona en git; mantenerlo < 5 MB (con una prueba que lo vigile) y
  documentar el compromiso entre tamaño y rendimiento cuando se reduzca un ensamble.
- **Control de mutaciones**: al terminar las pruebas de un modelo, romper a propósito cada protección (umbral de
  aviso, límites de `Entrada`, estratificación, imputación, texto) y comprobar que alguna prueba falla. Las
  pruebas que solo leen el artefacto guardado no detectan cambios en `train.py`: añadir también pruebas
  estructurales sobre `candidatos()` / `dividir()`. Así se detectaron 8 pruebas débiles en el modelo 03.
- Para avisar de extrapolación usar `core.modelos.fuera_de_rango` y guardar `rango=` en el artefacto (el modelo 08
  aún tiene una copia local de esa lógica). La matriz de confusión se pide con `metricas_clasificacion(..., orden=...)`.
- Cada modelo se entrega en su rama `feature/modelo-XX-slug` con PR; un subagente lo verifica de forma
  independiente (pruebas, fuga de información, métricas reproducibles, contrato, coherencia de documentos).

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
| Modelo 08 grasa corporal     | ✅ entrenado (Lasso, R² prueba 0.557 / CV 0.695); falta interfaz |
| Modelo 09 aguacate           | ✅ entrenado (GB, R² prueba 0.418 temporal / CV 0.509); falta interfaz |
| Modelo 03 vino               | ✅ entrenado (RF, F1 macro prueba 0.591 / CV 0.595); falta interfaz |
| Modelos 01, 04–07, 10        | ⏳ plantillas con TODO; `dataset.csv` de los 10 ya está en su carpeta (verificado) |
| Voz a texto (Google)         | ⏳ endpoint valida archivo, responde 501                               |
| Emociones                    | ⏳ responde 501. Decidido y avisado al profesor (2026-10-09): Azure detecta el rostro, Google Vision da la emoción. **Condición del profesor: poder justificarlo en el documento** (`docs_latex/SPEC.md`) |
| Comandos de voz → modelo     | 🟡 reconoce el modelo; faltan parámetros y el tono según la emoción   |
| Interfaz Jarvis (React)      | ⏳ scaffold de Vite + cliente API + mocks                              |
| Documento LaTeX              | ⏳ esqueleto en `docs_latex/main.tex`                                  |

## Documentación científica

Al redactar resultados de modelos: tono académico (tercera persona, justificación formal con referencias),
en Markdown fácil de pasar a LaTeX/Overleaf, dentro del `analisis.md` de cada modelo.

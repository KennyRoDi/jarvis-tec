# JarvisTEC-Core — Memoria del proyecto (ADD)

Asistente personal de escritorio estilo "Jarvis" (Proyecto IA, I Semestre 2026, TEC San Carlos).
Este archivo es la memoria persistente de los agentes: si cambias una convención, comando o el estado
del proyecto, actualízalo aquí en el mismo commit.

## ▶ Reanudar el trabajo (estado al 2026-10-10)

El desarrollo de los modelos se pausó por límite de cuota. **Antes de seguir, leer este apartado y el de "Orden de desarrollo".**

**Estado de los PR** (apilados; ninguno está fusionado, el usuario aún no los ha aprobado; **no fusionar ni subir a `main` sin que lo pida**):

| PR | Rama                              | Base                              | Modelo            |
|----|-----------------------------------|-----------------------------------|-------------------|
| #1 | `feature/modelo-08-grasa-corporal`| `main`                            | 08 grasa corporal |
| #2 | `feature/modelo-09-aguacate`      | `feature/modelo-08-grasa-corporal`| 09 aguacate       |
| #3 | `feature/modelo-03-vino`          | `feature/modelo-09-aguacate`      | 03 vino           |
| #4 | `feature/modelo-04-churn`         | `feature/modelo-03-vino`          | 04 churn          |
| #5 | `feature/modelo-05-acv`           | `feature/modelo-04-churn`         | 05 acv            |
| #6 | `feature/modelo-06-hepatitis`     | `feature/modelo-05-acv`           | 06 hepatitis      |
| #7 | `feature/modelo-07-cirrosis`      | `feature/modelo-06-hepatitis`     | 07 cirrosis       |
| #8 | `feature/modelo-01-bitcoin`       | `feature/modelo-07-cirrosis`      | 01 bitcoin        |
| #9 | `feature/modelo-10-sp500`         | `feature/modelo-01-bitcoin`       | 10 sp500          |

Fusionar en orden #1 → #9; tras cada fusión GitHub redirige el siguiente a `main` (o cambiar la base a mano). Si se
piden cambios en un PR intermedio, hay que rebasar las ramas siguientes.

**Pendiente, en este orden:**
1. **Los 10 modelos están entrenados y cada uno tiene su notebook de Colab** (sin fusionar). Falta la interfaz, voz, visión y LaTeX (punto 4).
2. **Re-verificar el 05 y el 06 con un subagente independiente**: tras su última revisión cambiaron (05: se quitó el `bmi`, ganó la
   regresión logística, umbrales 0.11 y 0.045; 06: el ALP volvió a entrar y se quitó el sexo); solo se comprobaron con pruebas y mutaciones propias.
3. **PR de seguimiento #10** (`feature/seguimiento-08-09-03`, base `feature/modelo-10-sp500`): lleva a los modelos 08, 09 y 03 lo aprendido
   después (`entrenar()` pura + prueba `reproduce`, `Entrada` estricta, `core.fuera_de_rango` y control de mutaciones; en el 03 se quitó
   `SVC(probability=True)`). **Queda pendiente** mover `serie.py`, copiado hoy en los modelos 01 y 10, a `core/`.
4. Interfaz (Dev B), voz y visión, y documento LaTeX siguen sin empezar; ver `specs/alcance_spec.md`.

**Procedimiento por modelo** (el que se siguió en 08–05):
1. Explorar los datos: duplicados, nulos, fuga de información, si faltan valores que predicen el objetivo.
2. Si el formulario sería largo, experimento de cuántas variables hacen falta (guardarlo en `metricas.json`).
3. `train.py` con `entrenar()` pura (sin escribir en disco); `router.py` con `Entrada` estricta y avisos.
4. Pruebas: datos, entrada, modelo, la de `reproduce` y control de mutaciones (`herramientas/mutar.py`).
5. `analisis.md` con las 6 etapas y referencias **verificadas por búsqueda web**; añadirlas a `docs_latex/referencias.bib`.
6. `python3 herramientas/cerrar_modelo.py ...` (ver `herramientas/README.md`) y revisar el `git diff`.
7. Commits separados (core / modelo / docs), subagente verificador **de solo lectura** (darle la lista de mutaciones ya
   probadas para que busque otras), corregir sus hallazgos, PR apilado con la verificación descrita.

**Entorno:** `cd backend && ../venv/bin/pytest -q` (≈ 45 s; con `-m "not reproduce"` es mucho más rápido, pero **no excluir
`reproduce` en CI**). No hay Colab conectado: los modelos son pequeños y entrenan en segundos localmente.

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
| 4  | 04 churn           | Binaria, muchas categóricas, `TotalCharges` sucia, formulario de 9 campos elegidos de 18 | ✅ (2026-10-09) |
| 5  | 05 acv             | 4.9 % de positivos: métricas distintas a accuracy, umbrales, bmi faltante informativo | ✅ (2026-10-09) |
| 6  | 06 hepatitis       | Multiclase muy desbalanceada, clase de 7 filas | ✅ (2026-10-09) |
| 7  | 07 cirrosis        | 418 filas, 106 casi vacías, fuga de información, 4 etapas desiguales | ✅ (2026-10-09) |
| 8  | 01 bitcoin         | Serie temporal: fechas, `-`, partición temporal, rezagos, predicción recursiva | ✅ (2026-10-10) |
| 9  | 10 sp500           | Serie temporal multi-símbolo; interpretar el símbolo desde la voz | ✅ (2026-10-10) |

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
- **Imputación + árboles = fuga silenciosa** (modelo 05): si el valor faltante predice el objetivo, imputar con la
  mediana deja un pico que un árbol aísla y "aprende" el artefacto sin que nadie lo incluya (Random Forest con `bmi`
  imputado: AUC 0.8455; sin `bmi`: 0.8347). Comprobarlo reajustando el modelo **con y sin** la variable; si el aporte
  desaparece, excluir la variable. Los umbrales y el recall con pocos positivos dependen de la semilla: reportar el rango.
- **Medir antes de afirmar una fuga** (modelo 07): `N_Days`/`Status` son seguimiento posterior y se excluyeron por principio, pero
  agregarlos no inflaba la métrica (0.395 contra 0.398). Excluir por principio y decir lo medido; no escribir "fuga" sin evidencia.
- **Métricas ordinales** (etapas): kappa cuadrático, error medio y aciertos a ±1; la línea base "siempre la categoría central" gana en las dos
  últimas, así que no deben leerse aisladas. Probar siempre la línea base con cada métrica.
- **Pacientes incompletos**: probar entrenar solo con completos frente a añadir los incompletos imputados en los **mismos pliegues** de completos.
- **Describir los datos no es medir lo que el modelo usa** (modelo 07): las medianas por etapa mostraban bilirrubina, albúmina y signos
  "claros", pero al quitar cada variable ninguna aportaba más de 0.026 de F1. Incluir siempre un experimento "quitar cada variable"
  antes de afirmar qué señales importan.
- **Documentar la población de cada cifra** (todos los pacientes / solo los completos) y que coincida con la de las figuras; citar el recall
  por clase **en la prueba, fuera de muestra y con otras particiones** (el de la etapa 4: 0.32, 0.73 y 0.58–0.74) y decir cuál es la publicada.
- **Probar las unidades y descripciones del formulario** (una prueba por campo): se corrigieron SGOT (U/L) y plaquetas (10³/µL).
- Probar también `entender()` y `explorar()` (con `capsys` y `tmp_path`): sin pruebas, mutarlas pasa inadvertido.
- **Artefacto contra señal** (modelos 05 y 06): ante un valor faltante sospechoso, reajustar con (a) la variable quitada, (b) los vacíos
  rellenados **al azar con valores observados** y (c) solo el indicador de faltante. Si el aporte desaparece con (b), es un artefacto
  (el `bmi` del 05); si sobrevive, es señal (el ALP del 06). No concluir por analogía: el 06 lo hizo y la revisión lo corrigió.
- **`herramientas/mutar.py` restaura ambos archivos antes de cada mutación** (un fallo anterior contaminaba las mutaciones consecutivas).
- **Series de precios (modelo 01)**: comparar siempre con la **persistencia** (mañana = hoy) y con la deriva; predecir el retorno
  logarítmico (no el precio) con características solo de cierres pasados para poder encadenar la predicción recursiva; medir la
  habilidad con IC (bootstrap por bloques) y **el acierto de dirección contra "siempre sube"** (en un mercado alcista coinciden). Si ningún modelo
  supera a la persistencia, decirlo en el análisis y en el `texto` de la API, y ofrecer lo que sí funciona (el intervalo, verificando su cobertura por mitades de la prueba). Un Ridge con α
  en el borde de la rejilla se reduce a su intercepto: reportarlo.
- **Clases raras (< 30 casos) en multiclase** (modelo 06): ponderar las clases y declarar que los puntajes no son probabilidades
  calibradas; reportar IC bootstrap de la prueba y las predicciones fuera de muestra del entrenamiento (más estables por
  clase); mirar la vista binaria enfermedad/sano. Verificar si el origen de las clases es distinto (los donantes del 06 tienen ≥ 32 años).
- **`entrenar()` pura + prueba `reproduce`**: separar el entrenamiento (selección, umbral, evaluación) de la escritura
  en disco, y añadir una prueba marcada `@pytest.mark.reproduce` que lo reejecute en memoria y exija igualdad
  exacta con `metricas.json`, el umbral, el rango y las probabilidades del artefacto. Es lo que atrapa fugas y
  errores dentro del entrenamiento (en el modelo 04 sobrevivían 44 de 74 mutaciones sin ella). Excluir en
  desarrollo con `pytest -m "not reproduce"`. Lo tienen los modelos 03 a 10.
- `Entrada` estricta (`ConfigDict(extra="forbid", strict=True)`): un nombre de campo mal escrito o `"12"` por `12` no
  deben aceptarse en silencio. Con una fecha, `strict=True` rechaza hasta "2017-09-15" (pydantic valida el cuerpo en modo Python) y el modo laxo
  acepta enteros como marcas de tiempo: usar `Field(strict=False)` con un `field_validator(mode="before")` que admita solo `AAAA-MM-DD` (modelo 09).
- Un test transversal comprueba que cada comando de voz se asocia a su propio modelo (`tests/test_api.py`).
- Los textos para voz no deben contener a la vez las dos conclusiones ("riesgo alto" en un texto de riesgo bajo).
- **Pruebas de límites con valores literales**: leer los límites desde `Entrada.model_fields` hace que la prueba cambie junto con el código
  (en el 08, 7 mutaciones de límites sobrevivieron así). Escribir los límites esperados en la prueba. Una prueba de "fecha laxa" debe usar un
  valor que pasaría de verdad (un entero que sea medianoche UTC y esté dentro del rango), no uno que otra regla ya rechaza.
- **Una prueba de "no usa la prueba" no puede alterar las etiquetas** si la partición es estratificada (cambia la partición): alterar las
  medidas de las filas de prueba. Y no ejecutar `pytest` en una carpeta mientras `mutar.py` la está mutando (lee el código mutado).
- `SVC(probability=...)` vale `"deprecated"` por defecto en scikit-learn 1.9: probar `hasattr(pipeline, "predict_proba")`, no el parámetro.
  Si la API devuelve probabilidades, la selección solo considera candidatos que las calculan (`elegibles()` en el 03).
- **Control de mutaciones**: al terminar las pruebas de un modelo, romper a propósito cada protección (umbral de
  aviso, límites de `Entrada`, estratificación, imputación, texto) y comprobar que alguna prueba falla. Las
  pruebas que solo leen el artefacto guardado no detectan cambios en `train.py`: añadir también pruebas
  estructurales sobre `candidatos()` / `dividir()`. Así se detectaron 8 pruebas débiles en el modelo 03.
- Para avisar de extrapolación usar `core.modelos.fuera_de_rango` y guardar `rango=` en el artefacto (los modelos 03 a 10 la usan). La matriz de confusión se pide con `metricas_clasificacion(..., orden=...)`.
- **`cross_val_score` con etiquetas de texto**: el scorer `average_precision` falla y devuelve `nan` en silencio.
  Usar `make_scorer(average_precision_score, response_method="predict_proba", pos_label=...)` y siempre
  `error_score="raise"`.
- Desbalance: sin `class_weight` si se muestran probabilidades; elegir el umbral con
  `core.entrenamiento.umbral_optimo_f1` sobre predicciones fuera de muestra (`cross_val_predict`) del
  entrenamiento, guardarlo en el artefacto (`umbral=`) y usarlo en el router (no el 0.5 por defecto).
- Reducir variables con evidencia: comparar AUC/F1 de validación cruzada con 18 / 9 / 6 / 3 variables y guardar el
  experimento en `metricas.json`; un formulario corto vale la pena si la pérdida es menor que la desviación.
- **Pocos positivos en la prueba** (<100): acompañar las métricas con intervalos bootstrap
  (`core.entrenamiento.intervalo_bootstrap`) y ofrecer un umbral de sensibilidad (`umbral_para_recall`) además del de
  F1; con el umbral 0.5 la exactitud puede igualar la de predecir siempre "No" (ACV: 95.1 %) sin detectar a nadie.
- **Datos faltantes**: comprobar si el valor faltante predice el objetivo (en ACV, `bmi` nulo: 19.9 % contra 4.3 %); si es
  un artefacto de la recolección, imputar en el `Pipeline` y no usar un indicador que la aplicación no puede dar.
- **Higiene de pruebas**: nunca `assert ... or True` ni aserciones que no puedan fallar; si un valor válido no puede
  disparar una rama (p. ej. el aviso de rango del `bmi`), dejar esa imposibilidad como prueba explícita. Probar los
  umbrales en sus **bordes** con las filas reales más cercanas a cada lado (un umbral de 0.07 en lugar de 0.06 no se
  detecta con filas lejanas). Los mutantes equivalentes (mismo comportamiento con los datos) se declaran, no se fuerzan.
- Comprobar si el dataset es real o de ejemplo (el de churn es una muestra ficticia de IBM) y decirlo en las limitaciones.
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
| Modelo 04 churn              | ✅ entrenado (RF, AUC prueba 0.840 / CV 0.845, umbral 0.35); falta interfaz |
| Modelo 05 acv                | ✅ entrenado (regresión logística, 4 variables sin bmi; AUC prueba 0.840 / CV 0.842, umbrales 0.11 y 0.045); falta interfaz |
| Modelo 06 hepatitis          | ✅ entrenado (RF, F1 macro prueba 0.580 / CV 0.635, 11 variables con ALP y sin sexo); falta interfaz |
| Modelo 07 cirrosis           | ✅ entrenado (RF, F1 macro prueba 0.465 / CV 0.460, solo pacientes completos); falta interfaz |
| Modelo 01 bitcoin            | ✅ entrenado (Ridge ≈ deriva, sin mejora demostrable sobre la persistencia; intervalo 95 % con cobertura 96–97 %); falta interfaz |
| Modelo 10 sp500              | ✅ entrenado (Ridge ≈ deriva compartido por 4 símbolos; su ventaja sobre la persistencia es solo la deriva); falta interfaz |
| Voz a texto (Google)         | ⏳ endpoint valida archivo, responde 501                               |
| Emociones                    | ⏳ responde 501. Decidido y avisado al profesor (2026-10-09): Azure detecta el rostro, Google Vision da la emoción. **Condición del profesor: poder justificarlo en el documento** (`docs_latex/SPEC.md`) |
| Comandos de voz → modelo     | 🟡 reconoce el modelo; faltan parámetros y el tono según la emoción   |
| Interfaz Jarvis (React)      | ⏳ scaffold de Vite + cliente API + mocks                              |
| Documento LaTeX              | ⏳ esqueleto en `docs_latex/main.tex`                                  |

## Documentación científica

Al redactar resultados de modelos: tono académico (tercera persona, justificación formal con referencias),
en Markdown fácil de pasar a LaTeX/Overleaf, dentro del `analisis.md` de cada modelo.

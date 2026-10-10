# SPEC — Modelo 06 · Clasificación del tipo de hepatitis C

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Qué categoría presenta el paciente: donante, hepatitis, fibrosis o cirrosis?
- **Tipo:** Clasificación multiclase
- **Variable objetivo:** `Category`

## Datos

- **Fuente:** https://www.kaggle.com/fedesoriano/hepatitis-c-dataset (`HepatitisCdata.csv`). ⚠️ En el enunciado este enlace está intercambiado con el de grasa corporal
- **Archivo:** `dataset.csv` en esta carpeta.
- ✅ Verificado: 615 filas. Hay una columna `Unnamed: 0` (índice): descartarla. Valores de `Category`: `0=Blood Donor` 533, `3=Cirrhosis` 30, `1=Hepatitis` 24, `2=Fibrosis` 21, `0s=suspect Blood Donor` 7.
- Variables: `Age, Sex` y análisis de laboratorio `ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT`.
- ✅ Clase `0s=suspect Blood Donor` (7 filas): **se excluye** (valores claramente anormales, albúmina mediana 21.6 g/L; no es donante ni etapa de la enfermedad).
- ✅ **Los 18 vacíos de `ALP` están todos en pacientes** (0 en donantes): artefacto. Con la mediana como imputación el Random Forest lo aprovecha (F1 macro 0.642 con ALP, 0.588 sin ella); la regresión logística no (0.582 / 0.576). **`ALP` se excluye.**
- ✅ Ningún donante tiene menos de 32 años (los pacientes llegan a 19): riesgo de aprender la población y no la enfermedad.
- Las unidades de los análisis no están documentadas en la fuente: se infirieron por el rango (g/L, U/L, µmol/L, mmol/L, kU/L); confirmarlas.
- Hay 31 nulos en 5 columnas de laboratorio (`ALP` 18, `CHOL` 10). Tras excluir la clase sospechosa: 533 donantes (88 %), 30 cirrosis, 24 hepatitis y 21 fibrosis.

## Enfoque sugerido

- Partición estratificada (la prueba tiene solo 4–6 casos por clase de enfermedad). Candidatos evaluados: regresión logística, k vecinos y Random Forest (ganó Random Forest; empate dentro del ruido). **No se usó SVM**: `SVC(probability=True)` está deprecado en scikit-learn 1.9.
- Métrica: F1 macro más matriz de confusión, con intervalos bootstrap y predicciones fuera de muestra del entrenamiento (más estables por clase). Las clases se ponderan: los puntajes **no** son probabilidades calibradas.

## Contrato del endpoint

- `POST /api/modelos/hepatitis/predecir` · `GET /api/modelos/hepatitis/info`
- **Entrada (`Entrada` en `router.py`):** ✅ 11 campos obligatorios: `age`, `sex` (`f`/`m`) y 9 análisis (`alb`, `alt`, `ast`, `bil`, `che`, `chol`, `crea`, `ggt`, `prot`; **sin `alp`**). Estricta (`extra="forbid"`, `strict=True`).
- **Salida:** `prediccion` ∈ `donante`/`hepatitis`/`fibrosis`/`cirrosis` más los puntajes de las 4 clases en `probabilidades`. El `texto` los lista, aclara que no son probabilidades calibradas ni un diagnóstico, y avisa si un valor sale del rango de entrenamiento.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "tipo de hepatitis"
- "estado del higado"
- "diagnóstico de hepatitis"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Resultado (2026-10-09)

Random Forest (343 KB), 11 variables (sin `ALP`, artefacto), clases ponderadas. Prueba (122 personas, solo 4–6 casos por clase de enfermedad): exactitud 0.934, F1 macro 0.640 (IC 95 % 0.41–0.82) frente a 0.234; especificidad 1.0 y sensibilidad 0.67 (enfermedad contra donante); cirrosis se reconoce (recall 0.75 fuera de muestra) y hepatitis casi no (0.11). Las poblaciones de origen difieren (donantes ≥ 32 años). No es un diagnóstico. Detalle en `analisis.md`.

## Archivos

| Archivo          | Contenido                                                                |
|------------------|--------------------------------------------------------------------------|
| `dataset.csv`    | Datos                                                                    |
| `train.py`       | 6 etapas de la rúbrica → genera `modelo.joblib`, `metricas.json`, `figuras/` |
| `router.py`      | `MODELO_INFO`, `Entrada` y `POST /predecir`                              |
| `test_modelo.py` | Pruebas del endpoint con el modelo real                                  |
| `analisis.md`    | Redacción académica de las 6 etapas (pasa a LaTeX)                       |

## Referencias

Verificadas y listadas al final de `analisis.md`; están en `docs_latex/referencias.bib`.

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [x] `dataset.csv` disponible y `python -m features.modelo_06_hepatitis.train` corre sin errores
- [x] Entendimiento y exploración: estadísticas impresas y al menos 2 figuras en `figuras/`
- [x] Modelo en un `Pipeline` (el mismo preprocesamiento en el entrenamiento y en la API)
- [x] Evaluación en el conjunto de prueba con las métricas de `specs/modelos_spec.md` y comparación con una línea base
- [x] `analisis.md` con las 6 secciones redactadas y al menos una referencia científica que justifique el algoritmo

**API REST**
- [x] `Entrada` con campos tipados y validados (sin `extra="allow"`)
- [x] `texto` de la respuesta en lenguaje natural, listo para que JARVIS lo lea
- [x] `test_modelo.py`: predicción válida (200) y entrada inválida (422)

**Aplicación**
- [ ] Se puede ejecutar desde la interfaz (formulario o comando de voz) y el resultado se muestra y se lee en voz alta

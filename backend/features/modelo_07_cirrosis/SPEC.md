# SPEC — Modelo 07 · Clasificación de la etapa de cirrosis

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿En qué etapa histológica (1–4) de cirrosis está el paciente?
- **Tipo:** Clasificación multiclase
- **Variable objetivo:** `Stage`

## Datos

- **Fuente:** https://www.kaggle.com/fedesoriano/cirrhosis-prediction-dataset (`cirrhosis.csv`). Proviene del ensayo de la Clínica Mayo sobre cirrosis biliar primaria
- **Archivo:** `dataset.csv` en esta carpeta.
- ✅ Verificado: 418 pacientes (`Stage` 1:21, 2:92, 3:155, 4:144, 6 nulos que se descartan → 412). 312 participaron en el ensayo aleatorizado y 106 no (sin ascitis, hepatomegalia, angiomas, cobre, fosfatasa alcalina ni SGOT: 100 de ellos con etapa).
- ✅ Los vacíos **no dependen de la etapa** (distribución de etapas casi idéntica en el bloque y fuera de él) y los pacientes incompletos **no mejoran el modelo**: se entrena solo con los **276 completos** (etapas 12, 59, 111 y 94).
- `Age` viene en días (9598–28650): convertir a años. `Edema` toma `Y`/`N`/`S`; `Status` toma `C`/`D`/`CL`.
- `N_Days` y `Status` describen el seguimiento posterior, no el estado del paciente al consultar: **se excluyen** (por principio). ✅ Medido: **no inflan** la métrica (con ellas 0.395, sin ellas 0.398 con la regresión logística; solas, 0.228 y 0.29–0.33 con el Random Forest): no afirmar una fuga que no se observó. `Drug` e `ID` también se excluyen.

## Enfoque sugerido

- Candidatos evaluados: regresión logística, Random Forest y Gradient Boosting (ganó Random Forest; diferencias dentro del ruido), con clases ponderadas (los puntajes no son probabilidades calibradas).
- Partición estratificada; F1 macro más métricas **ordinales** (kappa cuadrático, error medio y aciertos a una etapa: la línea base "siempre etapa 3" gana en las dos últimas), IC bootstrap y predicciones fuera de muestra del entrenamiento.

## Contrato del endpoint

- `POST /api/modelos/cirrosis/predecir` · `GET /api/modelos/cirrosis/info`
- **Entrada (`Entrada` en `router.py`):** ✅ 15 campos obligatorios: `age` (años), `sex` (F/M), `ascites`/`hepatomegaly`/`spiders` (Y/N), `edema` (N/S/Y) y 9 análisis (`bilirubin`, `cholesterol`, `albumin`, `copper`, `alk_phos`, `sgot`, `tryglicerides`, `platelets`, `prothrombin`). Estricta (`extra="forbid"`, `strict=True`): rechaza `n_days`, `status`, `drug` e `id`.
- **Salida:** `prediccion` etapa entera 1–4 más los puntajes de las 4 etapas (`probabilidades` con claves "1"…"4"). El `texto` aclara que no es un diagnóstico (la etapa se determina por biopsia) y que los puntajes no son probabilidades calibradas, y avisa si un valor sale del rango de entrenamiento.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "etapa de cirrosis"
- "estadio de la cirrosis"
- "tipo de cirrosis"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Resultado (2026-10-09)

Random Forest (480 KB), 15 variables de la consulta inicial, entrenado solo con los 276 pacientes completos; `N_Days`/`Status` excluidos (medido: no inflan). Prueba (56 pacientes; solo 2 en la etapa 1): F1 macro 0.465 (IC 95 % 0.25–0.60) frente a 0.146; kappa cuadrático 0.34; fuera de muestra, F1 macro 0.48, kappa 0.53 y a una etapa de error en 93 %. La etapa 4 se reconoce (recall 0.73); las etapas 2 y 3 se confunden; la línea base 'siempre etapa 3' gana en error medio. No es un diagnóstico. Detalle en `analisis.md`.

## Archivos

| Archivo          | Contenido                                                                |
|------------------|--------------------------------------------------------------------------|
| `dataset.csv`    | Datos                                                                    |
| `train.py`       | 6 etapas de la rúbrica → genera `modelo.joblib`, `metricas.json`, `figuras/` |
| `router.py`      | `MODELO_INFO`, `Entrada` y `POST /predecir`                              |
| `test_modelo.py` | Pruebas del endpoint con el modelo real                                  |
| `analisis.md`    | Redacción académica de las 6 etapas (pasa a LaTeX)                       |
| `notebook.ipynb` | Versión didáctica para Google Colab (extra; no reemplaza a `train.py`)      |

## Referencias

Verificadas y listadas al final de `analisis.md`; están en `docs_latex/referencias.bib`.

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [x] `dataset.csv` disponible y `python -m features.modelo_07_cirrosis.train` corre sin errores
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

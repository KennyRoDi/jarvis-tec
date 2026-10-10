# SPEC — Modelo 05 · Clasificación de riesgo de accidente cerebrovascular

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿El paciente tiene riesgo de sufrir un accidente cerebrovascular?
- **Tipo:** Clasificación binaria (muy desbalanceada)
- **Variable objetivo:** `stroke` (0/1; ≈ 5 % positivos)

## Datos

- **Fuente:** https://www.kaggle.com/fedesoriano/stroke-prediction-dataset (`healthcare-dataset-stroke-data.csv`)
- **Archivo:** `dataset.csv` en esta carpeta.
- ✅ Verificado: 5110 filas, 249 positivos (4.9 %), 201 nulos reales en `bmi` (ya viene como número, no como texto `N/A`), 1 fila con `gender = Other`.
- Columnas: `id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke`.
- Descartar `id`. ✅ **El `bmi` faltante es informativo** (19.9 % de ACV con `bmi` nulo contra 4.3 % con él): es un artefacto que **se filtraba al Random Forest por la imputación con la mediana** (AUC 0.8455 con `bmi`, 0.8347 sin él). Por eso el `bmi` se excluye del modelo.
- ✅ El origen del conjunto está declarado como confidencial en Kaggle: documentarlo como limitación. `smoking_status = Unknown` ocurre en 80 % de los menores de 18 años (y 44 % de los `Unknown` son menores): significa "no registrado".

## Enfoque sugerido

- Un clasificador que siempre dice "no" logra ≈ 95 % de accuracy: **no usar accuracy como métrica principal**.
- Métricas: AUC ROC/PR, Brier, calibración, recall y F1 de la clase positiva, con **intervalos de confianza bootstrap** (solo 50 positivos en la prueba). **Sin `class_weight`** (la probabilidad se muestra al usuario): el desbalance se trata con dos umbrales calculados con predicciones fuera de muestra (F1 y sensibilidad ≥ 80 %).
- Se midió qué variables aportan: la edad sola da AUC 0.834 y las 4 elegidas, 0.842; las demás no aportan (el `bmi` solo aportaba un artefacto).
- No se usó SMOTE (`imbalanced-learn` no está en `requirements.txt`): el desbalance se trata con umbrales.
- Candidatos evaluados: regresión logística, Random Forest y Gradient Boosting (ganó la regresión logística; diferencias dentro del ruido).

## Contrato del endpoint

- `POST /api/modelos/acv/predecir` · `GET /api/modelos/acv/info`
- **Entrada (`Entrada` en `router.py`):** ✅ 4 campos obligatorios: `age`, `hypertension`, `heart_disease`, `avg_glucose_level` (se excluyen `bmi`, género, estado civil, trabajo, residencia y tabaquismo: no aportan). Estricta (`extra="forbid"`, `strict=True`).
- **Salida:** `prediccion` `"Yes"`/`"No"` (Yes = riesgo alto, umbral F1 0.11) más `probabilidades`. El `texto` indica el nivel (bajo < 0.045 ≤ moderado < 0.11 ≤ alto), advierte que no es un diagnóstico ni reemplaza a un profesional de la salud, y avisa si `age` o `avg_glucose_level` salen del rango de entrenamiento. La comparación con los umbrales usa la probabilidad redondeada a 4 decimales (la que se muestra).
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "riesgo de derrame"
- "riesgo de acv"
- "accidente cerebrovascular"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Resultado (2026-10-09)

Regresión logística (4 variables; el `bmi` se excluyó porque su aporte era un artefacto). Prueba (1 022 pacientes, 50 con ACV): AUC ROC 0.840 (IC 95 % 0.78–0.90), recall 0.64 con el umbral F1 (0.11) y 0.82 con el de sensibilidad (0.045); con el umbral 0.5 la exactitud es 95.1 % pero no detecta a nadie. Umbrales y recall sensibles a la semilla. No es un diagnóstico. Detalle en `analisis.md`.

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
- [x] `dataset.csv` disponible y `python -m features.modelo_05_acv.train` corre sin errores
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

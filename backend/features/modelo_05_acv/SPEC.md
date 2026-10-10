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
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- ✅ Verificado: 5110 filas, 249 positivos (4.9 %), 201 nulos reales en `bmi` (ya viene como número, no como texto `N/A`), 1 fila con `gender = Other`.
- Columnas: `id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke`.
- Descartar `id`.

## Enfoque sugerido

- Un clasificador que siempre dice "no" logra ≈ 95 % de accuracy: **no usar accuracy como métrica principal**.
- Métricas: recall y F1 de la clase positiva, más ROC-AUC. Usar `class_weight="balanced"` y ajustar el umbral.
- SMOTE requiere `imbalanced-learn`, que no está en `requirements.txt`: acordarlo antes de agregarlo.
- Candidatos: Regresión logística y Random Forest.

## Contrato del endpoint

- `POST /api/modelos/acv/predecir` · `GET /api/modelos/acv/info`
- **Entrada (`Entrada` en `router.py`):** Las 10 variables clínicas (sin `id`).
- **Salida:** `prediccion` 0/1 más `probabilidades`. El `texto` habla de riesgo, no de diagnóstico.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "riesgo de derrame"
- "accidente cerebrovascular"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Archivos

| Archivo          | Contenido                                                                |
|------------------|--------------------------------------------------------------------------|
| `dataset.csv`    | Datos                                                                    |
| `train.py`       | 6 etapas de la rúbrica → genera `modelo.joblib`, `metricas.json`, `figuras/` |
| `router.py`      | `MODELO_INFO`, `Entrada` y `POST /predecir`                              |
| `test_modelo.py` | Pruebas del endpoint con el modelo real                                  |
| `analisis.md`    | Redacción académica de las 6 etapas (pasa a LaTeX)                       |

## Referencias sugeridas

- _Pendiente: al menos un artículo científico que justifique el algoritmo elegido._

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [ ] `dataset.csv` disponible y `python -m features.modelo_05_acv.train` corre sin errores
- [ ] Entendimiento y exploración: estadísticas impresas y al menos 2 figuras en `figuras/`
- [ ] Modelo en un `Pipeline` (el mismo preprocesamiento en el entrenamiento y en la API)
- [ ] Evaluación en el conjunto de prueba con las métricas de `specs/modelos_spec.md` y comparación con una línea base
- [ ] `analisis.md` con las 6 secciones redactadas y al menos una referencia científica que justifique el algoritmo

**API REST**
- [ ] `Entrada` con campos tipados y validados (sin `extra="allow"`)
- [ ] `texto` de la respuesta en lenguaje natural, listo para que JARVIS lo lea
- [ ] `test_modelo.py`: predicción válida (200) y entrada inválida (422)

**Aplicación**
- [ ] Se puede ejecutar desde la interfaz (formulario o comando de voz) y el resultado se muestra y se lee en voz alta

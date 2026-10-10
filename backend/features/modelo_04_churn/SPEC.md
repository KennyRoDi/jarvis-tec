# SPEC — Modelo 04 · Clasificación de abandono de clientes de telefonía

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿El cliente abandonará la compañía telefónica?
- **Tipo:** Clasificación binaria
- **Variable objetivo:** `Churn` (Yes/No; ≈ 26 % Yes)

## Datos

- **Fuente:** https://github.com/IBM/telco-customer-churn-on-icp4d ✅ ya descargado
- **Archivo:** `dataset.csv` en esta carpeta.
- ✅ Verificado: 7043 filas, `Churn` = Yes en 26.5 %. Se descarta `customerID`.
- `TotalCharges` es texto con 11 cadenas vacías (todas de clientes con `tenure` = 0, ✅ verificado) y es redundante (correlación 0.9996 con `tenure × MonthlyCharges`): **se descarta**.
- ✅ Es un conjunto de ejemplo **ficticio** de IBM Cognos Analytics: documentarlo como limitación.
- ✅ 40 filas repiten las 19 variables de otra (clientes distintos, abandono posiblemente distinto): se conservan. Sin internet, los 6 servicios adicionales valen "No internet service"; sin teléfono, `MultipleLines` vale "No phone service".
- La mayoría de las columnas son categóricas (Yes/No, tipo de contrato, método de pago).

## Enfoque sugerido

- Candidatos: Regresión logística, Random Forest y Gradient Boosting.
- Clases desbalanceadas: se reporta AUC ROC/PR, Brier, recall y F1 de `Yes` (no solo accuracy). **Sin `class_weight`** (las probabilidades se muestran al usuario y deben estar calibradas); el desbalance se trata con un umbral elegido con predicciones fuera de muestra del entrenamiento (`core.entrenamiento.umbral_optimo_f1`).
- Se midió cuántas variables hacen falta (18 / 9 / 6 / 3): 9 pierden solo 0.002 de AUC.
- Línea base: `DummyClassifier`.

## Contrato del endpoint

- `POST /api/modelos/churn/predecir` · `GET /api/modelos/churn/info`
- **Entrada (`Entrada` en `router.py`):** ✅ 9 campos obligatorios: `tenure`, `monthly_charges`, `contract`, `internet_service`, `payment_method`, `paperless_billing`, `tech_support`, `online_security`, `senior_citizen` (valores como en el dataset). Entrada estricta (`extra="forbid"`, `strict=True`): rechaza campos desconocidos y tipos laxos. Se rechaza la combinación incoherente internet/servicios.
- **Salida:** `prediccion` `"Yes"`/`"No"` según el **umbral guardado en el artefacto** (0.35, no 0.5) más `probabilidades` de ambas. `texto` dice riesgo alto o bajo (una sola vez, sin ambigüedad al oírlo) y avisa si `tenure`/`monthly_charges` están fuera del rango de entrenamiento.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "cliente se va"
- "se va a pasar de compania"
- "abandono de cliente"
- "churn"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Resultado (2026-10-09)

Random Forest (1 MB), 9 variables (AUC solo 0.002 menor que con 18), umbral 0.35 elegido con predicciones fuera de muestra. Prueba (1 409 clientes): AUC ROC 0.840, recall de los que se van 0.69 (0.53 con umbral 0.5), precisión 0.55, probabilidades calibradas (Brier 0.138 frente a 0.195). Conjunto ficticio de IBM: limitación documentada. Detalle en `analisis.md`.

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
- [x] `dataset.csv` disponible y `python -m features.modelo_04_churn.train` corre sin errores
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

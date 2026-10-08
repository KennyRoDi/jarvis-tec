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
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- 7043 filas. Se descarta `customerID`.
- `TotalCharges` es texto y tiene cadenas vacías (clientes con `tenure` = 0): convertir a número e imputar.
- La mayoría de las columnas son categóricas (Yes/No, tipo de contrato, método de pago).

## Enfoque sugerido

- Candidatos: Regresión logística, Random Forest y Gradient Boosting.
- Clases desbalanceadas: reportar recall y F1 de la clase `Yes` (no solo accuracy); considerar `class_weight="balanced"`.
- Línea base: `DummyClassifier`.

## Contrato del endpoint

- `POST /api/modelos/churn/predecir` · `GET /api/modelos/churn/info`
- **Entrada (`Entrada` en `router.py`):** Subconjunto de variables relevantes (p. ej. `tenure`, `monthly_charges`, `contract`, `internet_service`, `payment_method`, …). Definirlo y documentarlo aquí.
- **Salida:** `prediccion` `"Yes"`/`"No"` más `probabilidades`.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "cliente se va"
- "abandono de cliente"
- "churn"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_04_churn.train` corre sin errores
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

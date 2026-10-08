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
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- Variables esperadas: `Age, Sex` y análisis de laboratorio `ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT`.
- Valores de `Category`: `0=Blood Donor`, `0s=suspect Blood Donor`, `1=Hepatitis`, `2=Fibrosis`, `3=Cirrhosis`. Decidir y justificar qué hacer con `0s`.
- Hay nulos en varias columnas de laboratorio. Las clases están muy desbalanceadas (mayoría de donantes).

## Enfoque sugerido

- Partición estratificada; candidatos KNN (con escalado), Random Forest y SVM.
- Métrica: F1 macro más matriz de confusión.

## Contrato del endpoint

- `POST /api/modelos/hepatitis/predecir` · `GET /api/modelos/hepatitis/info`
- **Entrada (`Entrada` en `router.py`):** `age`, `sex` y los 10 análisis de laboratorio.
- **Salida:** `prediccion` categoría legible (sin el prefijo numérico) más `probabilidades`.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "tipo de hepatitis"
- "diagnóstico de hepatitis"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_06_hepatitis.train` corre sin errores
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

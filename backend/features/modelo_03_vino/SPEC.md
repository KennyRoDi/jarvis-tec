# SPEC — Modelo 03 · Clasificación de la calidad del vino

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿El vino es de calidad baja, media o alta según sus propiedades fisicoquímicas?
- **Tipo:** Clasificación multiclase
- **Variable objetivo:** `quality` (3–9) agrupada en clases. Propuesta: baja ≤ 5, media = 6, alta ≥ 7

## Datos

- **Fuente:** https://www.kaggle.com/rajyellow46/wine-quality (`winequalityN.csv`)
- **Archivo:** `dataset.csv` en esta carpeta (390 KB).
- ✅ Verificado: 6497 filas. Las columnas vienen con espacios (`fixed acidity`, `free sulfur dioxide`…): renombrar a `snake_case` en `train.py`. `quality` = 3:30, 4:216, 5:2138, 6:2836, 7:1079, 8:193, 9:5.
- Variables: `type` (red/white, 1599/4898) más 11 fisicoquímicas (acidez, azúcar, cloruros, sulfitos, densidad, pH, sulfatos, alcohol).
- Hay 38 nulos repartidos en 7 columnas (✅ verificado): imputar dentro del `Pipeline`.
- ✅ **1 168 filas duplicadas (18 %)**: se descartan antes de dividir; no hacerlo infla la exactitud ~10 puntos (experimento en `metricas.json`).
- Las clases extremas son escasas: por eso se agrupan, y la partición debe ser estratificada.

## Enfoque sugerido

- Candidatos evaluados: regresión logística, SVM, Random Forest y Gradient Boosting (ganó Random Forest).
- Métrica principal: F1 macro, más la matriz de confusión.
- Línea base: `DummyClassifier` (clase más frecuente).

## Contrato del endpoint

- `POST /api/modelos/vino/predecir` · `GET /api/modelos/vino/info`
- **Entrada (`Entrada` en `router.py`):** ✅ 12 campos obligatorios y validados: `tipo` (`red`/`white`) y las 11 variables fisicoquímicas en `snake_case` (`fixed_acidity`, …, `ph`, `sulphates`, `alcohol`).
- **Salida:** `prediccion` clase (`baja`/`media`/`alta`) más `probabilidades` de las tres. `texto` avisa si ninguna supera el 50 %.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`). La API exige los 12 datos y rechaza
`free_sulfur_dioxide > total_sulfur_dioxide`; avisa si una medida está fuera del rango de entrenamiento.

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "calidad del vino"
- "clasificar vino"
- "que tan bueno es el vino"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Resultado (2026-10-09)

Random Forest compacto (100 árboles, 2.9 MB). Prueba (1 066 filas): exactitud 0.594, F1 macro 0.591 frente a 0.437 y 0.203 de la clase mayoritaria; consistente con la validación cruzada (0.595). 94 % de los errores son entre clases vecinas. Eliminar las 1 168 filas duplicadas evita inflar la exactitud ~10 puntos. Detalle en `analisis.md`.

## Archivos

| Archivo          | Contenido                                                                |
|------------------|--------------------------------------------------------------------------|
| `dataset.csv`    | Datos                                                                    |
| `train.py`       | 6 etapas de la rúbrica → genera `modelo.joblib`, `metricas.json`, `figuras/` |
| `router.py`      | `MODELO_INFO`, `Entrada` y `POST /predecir`                              |
| `test_modelo.py` | Pruebas del endpoint con el modelo real                                  |
| `analisis.md`    | Redacción académica de las 6 etapas (pasa a LaTeX)                       |
| `notebook.ipynb` | Versión didáctica para Google Colab (extra; no reemplaza a `train.py`)      |

## Referencias sugeridas

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [x] `dataset.csv` disponible y `python -m features.modelo_03_vino.train` corre sin errores
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

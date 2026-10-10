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
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- ✅ Verificado: 6497 filas. Las columnas vienen con espacios (`fixed acidity`, `free sulfur dioxide`…): renombrar a `snake_case` en `train.py`. `quality` = 3:30, 4:216, 5:2138, 6:2836, 7:1079, 8:193, 9:5.
- Variables: `type` (red/white, 1599/4898) más 11 fisicoquímicas (acidez, azúcar, cloruros, sulfitos, densidad, pH, sulfatos, alcohol).
- Hay 38 nulos repartidos en 7 columnas (✅ verificado): imputar dentro del `Pipeline`.
- Las clases extremas son escasas: por eso se agrupan, y la partición debe ser estratificada.

## Enfoque sugerido

- Candidatos: Random Forest, SVM (con escalado) y Regresión logística.
- Métrica principal: F1 macro, más la matriz de confusión.
- Línea base: `DummyClassifier` (clase más frecuente).

## Contrato del endpoint

- `POST /api/modelos/vino/predecir` · `GET /api/modelos/vino/info`
- **Entrada (`Entrada` en `router.py`):** Las 12 variables en `snake_case` (`fixed_acidity`, …, `alcohol`, `type`).
- **Salida:** `prediccion` clase (`baja`/`media`/`alta`) más `probabilidades`.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "calidad del vino"
- "clasificar vino"

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

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [ ] `dataset.csv` disponible y `python -m features.modelo_03_vino.train` corre sin errores
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

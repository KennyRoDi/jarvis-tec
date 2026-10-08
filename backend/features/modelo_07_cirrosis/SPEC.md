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
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- ≈ 418 pacientes; ≈ 106 no participaron en el ensayo y tienen muchas columnas vacías. `Stage` también tiene nulos: descartar esas filas.
- `Age` viene en días: convertir a años.
- `N_Days` y `Status` describen el seguimiento posterior, no el estado del paciente al consultar: **excluirlos** para evitar fuga de información.

## Enfoque sugerido

- Candidatos: Random Forest, Gradient Boosting y Regresión logística multinomial.
- Partición estratificada; métrica F1 macro.

## Contrato del endpoint

- `POST /api/modelos/cirrosis/predecir` · `GET /api/modelos/cirrosis/info`
- **Entrada (`Entrada` en `router.py`):** Variables clínicas y de laboratorio (`bilirubin`, `albumin`, `edema`, `ascites`, …).
- **Salida:** `prediccion` etapa (1–4) más `probabilidades`.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "etapa de cirrosis"
- "tipo de cirrosis"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_07_cirrosis.train` corre sin errores
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

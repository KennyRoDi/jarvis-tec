# SPEC — Modelo 09 · Predicción del precio del aguacate

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Cuál será el precio promedio del aguacate según la región, el tipo y la fecha?
- **Tipo:** Regresión
- **Variable objetivo:** `AveragePrice` (USD por unidad)

## Datos

- **Fuente:** https://www.kaggle.com/neuromusic/avocado-prices (`avocado.csv`)
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- Columnas esperadas: `Date, AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, …, type, year, region`.
- `region` mezcla ciudades con agregados (`TotalUS`, `West`, …): decidir y justificar si se usan.
- Los volúmenes son del mismo día que el precio y el usuario no los conoce de antemano: preferir `type`, `region`, año, mes y semana.

## Enfoque sugerido

- Candidatos: Random Forest y Gradient Boosting; línea base: media por región y tipo.
- Partición temporal o validación cruzada agrupada por fecha.

## Contrato del endpoint

- `POST /api/modelos/aguacate/predecir` · `GET /api/modelos/aguacate/info`
- **Entrada (`Entrada` en `router.py`):** `region` (enum), `type` (`conventional`/`organic`), `fecha`. Valores por defecto: `TotalUS`, `conventional`, hoy.
- **Salida:** `prediccion` número en USD.
- **¿Se ejecuta solo con la voz?** Sí: "precio del aguacate" se ejecuta con los valores por defecto.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "precio del aguacate"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_09_aguacate.train` corre sin errores
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

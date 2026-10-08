# SPEC — Modelo 01 · Predicción del precio del Bitcoin

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Cuál será el precio de cierre del Bitcoin en los próximos días?
- **Tipo:** Regresión (serie temporal)
- **Variable objetivo:** `Close` del día siguiente (se construye desplazando la serie)

## Datos

- **Fuente:** https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1 (archivo `bitcoin_price_Training - bitcoin_price.2013Apr-2017Aug.csv`)
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- Columnas esperadas: `Date, Open, High, Low, Close, Volume, Market Cap` (verificar al descargar).
- Las filas vienen de la más reciente a la más antigua: ordenar por fecha ascendente.
- `Volume` y `Market Cap` traen separadores de miles y `-` como faltante: limpiar antes de convertir a número.
- El dataset termina en agosto de 2017: "mañana" significa el día siguiente al último registro, y la respuesta debe decirlo.

## Enfoque sugerido

- Variables de rezago (`close_t-1 … close_t-k`), medias móviles y retornos.
- Partición **temporal** (sin `shuffle`) y validación con `TimeSeriesSplit`.
- Línea base: persistencia (el precio de mañana es igual al de hoy). Candidatos: Regresión lineal, Random Forest, Gradient Boosting.
- Para varios días adelante, predecir de forma recursiva.

## Contrato del endpoint

- `POST /api/modelos/bitcoin/predecir` · `GET /api/modelos/bitcoin/info`
- **Entrada (`Entrada` en `router.py`):** `dias_adelante: int = 1` (1–7). Todos los campos tienen valor por defecto, así que `{}` es válido.
- **Salida:** `prediccion` número en USD; `texto` incluye la fecha a la que corresponde.
- **¿Se ejecuta solo con la voz?** Sí: "JarvisTEC precio del bitcoin para mañana" se ejecuta sin formulario (`dias_adelante` = 1; "pasado mañana" = 2).

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "precio del bitcoin"
- "bitcoin mañana"
- "tipo de cambio del bitcoin"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_01_bitcoin.train` corre sin errores
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

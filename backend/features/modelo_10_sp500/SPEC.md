# SPEC — Modelo 10 · Predicción del precio de acciones del S&P 500

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Cuál será el precio de cierre de una acción del S&P 500 en los próximos días?
- **Tipo:** Regresión (serie temporal)
- **Variable objetivo:** `close` del día siguiente por símbolo

## Datos

- **Fuente:** https://www.kaggle.com/camnugent/sandp500 (`all_stocks_5yr.csv`)
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- ✅ Verificado: 619040 filas, 505 símbolos, del 2013-02-08 al 2018-02-07; 11 nulos en `open` y 8 en `high`/`low`.
- El archivo completo pesa ≈ 30 MB: el completo está en `data/all_stocks_5yr.csv` (fuera de git; se regenera con `bash data/descargar_datasets.sh`) y `dataset.csv` ya contiene el recorte de AAPL, MSFT, AMZN y GOOGL (5036 filas, 1259 por símbolo).
- "Mañana" significa el día siguiente al último registro del dataset (7-feb-2018).

## Enfoque sugerido

- Igual que el modelo 01: rezagos por símbolo, partición temporal y línea base de persistencia.
- Candidatos: Regresión lineal y Random Forest.

## Contrato del endpoint

- `POST /api/modelos/sp500/predecir` · `GET /api/modelos/sp500/info`
- **Entrada (`Entrada` en `router.py`):** `simbolo` (enum de los símbolos elegidos, por defecto `AAPL`) y `dias_adelante: int = 1`.
- **Salida:** `prediccion` número en USD; `texto` con el símbolo y la fecha.
- **¿Se ejecuta solo con la voz?** Sí: "precio de la acción de Apple para mañana" se ejecuta directo (extraer el símbolo del texto).

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "precio de la acción"
- "bolsa"
- "sp500"

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
- [ ] `dataset.csv` disponible y `python -m features.modelo_10_sp500.train` corre sin errores
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

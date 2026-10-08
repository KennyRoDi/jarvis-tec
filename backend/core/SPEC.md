# SPEC — Código compartido (`backend/core/`)

> **Responsables:** ambos devs, **solo de común acuerdo**: un cambio aquí afecta a todas las features.

## Módulos

| Módulo             | Ofrece                                                                                  |
|--------------------|-----------------------------------------------------------------------------------------|
| `config.py`        | Rutas, `HOST`/`PUERTO`, CORS y variables de `backend/.env`                              |
| `errores.py`       | `ApiError(status, codigo, mensaje, detalle)` y manejadores con el formato §1.1 de la spec |
| `modelos.py`       | Registro de modelos, `RespuestaPrediccion`, `predecir_con_pipeline`, `GET /api/modelos` y `/info` |
| `entrenamiento.py` | Métricas de regresión y clasificación, `guardar_figura`, `guardar_modelo` (joblib + metricas.json) |
| `escritorio.py`    | Concede cámara y micrófono a la ventana PyWebView (WebKitGTK)                           |

## Invariantes

- Toda respuesta de error sale con `{"error": {"codigo", "mensaje", "detalle"}}`.
- Un artefacto de modelo es `{"pipeline": Pipeline de sklearn, "metricas": dict, ...}`; el pipeline recibe un
  `DataFrame` con exactamente los campos de `Entrada`.
- `main.py` registra las features solo; ninguna función de `core` conoce una feature concreta.

## Cuándo cambiar `core`

Solo si algo se repite en 2 o más features o si lo exige la spec. Todo cambio va con su prueba en
`backend/tests/` y deja pasando la suite completa (`cd backend && pytest`).

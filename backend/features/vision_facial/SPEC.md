# SPEC — Visión facial

> **Responsable:** Dev B · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §5 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (A1)

## Objetivo

Agente inteligente que **reconoce el sentimiento de la persona a partir de un frame de la cámara en tiempo
real** (alcance A1). La emoción dominante alimenta la decisión del asistente (`/api/asistente/comando`).

## Proveedor: opción 1, Azure + Google Vision (decidida el 2026-10-08)

El enunciado exige Azure para reconocer rostros, pero **Microsoft retiró el atributo `emotion` de Face API**
en 2022. Por eso:

| Paso | Servicio | Archivo | Aporta |
|------|----------|---------|--------|
| 1 | Azure Face `detect_with_stream` (sin atributos) | `azure_face.py` | `rectangulo` de cada rostro ("Debe usar Azure para reconocer rostros") |
| 2 | Google Cloud Vision `face_detection` | `google_vision.py` | Probabilidad de alegría, tristeza, enojo y sorpresa |
| 3 | Combinación | `azure_face.detectar_emociones` | Lista `rostros` del contrato §5 |

Conversión de probabilidad a puntaje: `VERY_UNLIKELY` 0.0 · `UNLIKELY` 0.25 · `POSSIBLE` 0.5 · `LIKELY` 0.75 ·
`VERY_LIKELY` 1.0 (ya definida en `google_vision.py`). Vision cubre 4 emociones: `felicidad`, `tristeza`,
`enojo` y `sorpresa`; `miedo`, `desprecio` y `disgusto` quedan en 0.0, y `neutral` = 1 − máximo.

Los rostros de ambos servicios se emparejan por posición (ordenados por `x`). Si Google no detecta un rostro
que Azure sí, ese rostro va con `neutral` = 1.0.

Pendiente: informar la decisión al profesor (evidencia: error de Azure al pedir `emotion`).

## Tareas

1. Implementar `azure_face.py` (detección) y `google_vision.py` (emoción) según la tabla de arriba.
2. Mapear siempre a las 8 claves fijas de `EMOCIONES`; `emocion_dominante` = clave con mayor puntaje.
3. Errores del proveedor → `ApiError(502, "SERVICIO_EXTERNO", …)`. Sin rostros → `cantidad: 0` (no es error).
4. Respetar el límite del plan gratuito: la interfaz envía como máximo un frame cada 3–5 s.

## Diseño del agente (para el documento)

PEAS: **Rendimiento** = emoción correcta y a tiempo · **Entorno** = usuario frente a la webcam, iluminación
variable · **Actuadores** = emoción reportada al asistente y mostrada en pantalla · **Sensores** = cámara.
Tipo: agente reflejo simple (percepción → clasificación), cuyo resultado usa el agente de voz.

## Criterios de aceptación

- [x] Decisión del proveedor tomada y registrada en `specs/alcance_spec.md` y en `CLAUDE.md`
- [ ] Decisión informada al profesor
- [ ] Una foto real con un rostro devuelve `cantidad ≥ 1` y las 8 claves (prueba de integración con una
      imagen de muestra en `muestras/`, marcada con `skip` si no hay credenciales)
- [ ] Una foto sin rostros devuelve `cantidad: 0`
- [x] Formato inválido → 415; imagen vacía → 400 (`test_vision.py`)

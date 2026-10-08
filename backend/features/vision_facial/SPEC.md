# SPEC — Visión facial

> **Responsable:** Dev B · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §5 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (A1)

## Objetivo

Agente inteligente que **reconoce el sentimiento de la persona a partir de un frame de la cámara en tiempo
real** (alcance A1). La emoción dominante alimenta la decisión del asistente (`/api/asistente/comando`).

## ⚠️ Decisión pendiente: proveedor de emociones

El enunciado exige Azure para reconocer rostros, pero **Microsoft retiró el atributo `emotion` de Face API**
(anunciado en junio de 2022; los recursos nuevos no tienen acceso).

1. Crear el recurso Azure AI Face y llamar `detect_with_stream(..., return_face_attributes=["emotion"])`.
2. Si responde con emociones, implementar solo con Azure.
3. Si lo rechaza, **consultar al profesor** con la evidencia (mensaje de error). Alternativa propuesta, que
   el enunciado permite en esta entrega: Azure para detectar el rostro (`rectangulo`) y Google Cloud Vision
   `face_detection` para la emoción. Vision devuelve probabilidades cualitativas para alegría, tristeza,
   enojo y sorpresa; convertirlas así: `VERY_UNLIKELY` 0.0 · `UNLIKELY` 0.25 · `POSSIBLE` 0.5 ·
   `LIKELY` 0.75 · `VERY_LIKELY` 1.0. `neutral` = 1 − máximo; el resto de las claves, 0.0.

Con cualquiera de las dos opciones, **el contrato §5 no cambia**: la interfaz no se entera del proveedor.

## Tareas

1. Implementar `azure_face.py` (y, si aplica, un `google_vision.py` en esta carpeta) según la decisión.
2. Mapear siempre a las 8 claves fijas de `EMOCIONES`; `emocion_dominante` = clave con mayor puntaje.
3. Errores del proveedor → `ApiError(502, "SERVICIO_EXTERNO", …)`. Sin rostros → `cantidad: 0` (no es error).
4. Respetar el límite del plan gratuito: la interfaz envía como máximo un frame cada 3–5 s.

## Diseño del agente (para el documento)

PEAS: **Rendimiento** = emoción correcta y a tiempo · **Entorno** = usuario frente a la webcam, iluminación
variable · **Actuadores** = emoción reportada al asistente y mostrada en pantalla · **Sensores** = cámara.
Tipo: agente reflejo simple (percepción → clasificación), cuyo resultado usa el agente de voz.

## Criterios de aceptación

- [ ] Decisión del proveedor tomada y registrada en `specs/alcance_spec.md` y en `CLAUDE.md`
- [ ] Una foto real con un rostro devuelve `cantidad ≥ 1` y las 8 claves (prueba de integración con una
      imagen de muestra en `muestras/`, marcada con `skip` si no hay credenciales)
- [ ] Una foto sin rostros devuelve `cantidad: 0`
- [x] Formato inválido → 415; imagen vacía → 400 (`test_vision.py`)

# SPEC — Asistente de voz

> **Responsable:** Dev B · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §4 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (A2, A4 y "tomará una decisión")

## Objetivo

Agente inteligente que **recibe audio, lo convierte a texto y ejecuta la instrucción** (alcance A2). Además,
asocia comandos con los 10 modelos (A4) y adapta su respuesta a la emoción detectada por la cámara.

## Endpoints

| Endpoint                         | Archivo                   | Estado |
|----------------------------------|---------------------------|--------|
| `POST /api/voz/transcribir`      | `router.py` → `speech.py` | Valida el archivo; la transcripción responde 501 |
| `POST /api/asistente/comando`    | `router.py` → `comandos.py` | Reconoce el modelo y devuelve `emocion`; faltan los parámetros y el tono |

## Tareas

1. **Transcripción (`speech.py`)** con Google Cloud Speech-to-Text (obligatorio por el enunciado):
   - Credenciales en `GOOGLE_APPLICATION_CREDENTIALS` (`backend/.env`); nunca subir el JSON.
   - `audio/webm` del `MediaRecorder` del navegador → `WEBM_OPUS`; `audio/wav` → `LINEAR16`.
   - Idioma por defecto `es-CR`. Devolver la mejor alternativa y su `confianza`.
   - Los errores del proveedor se convierten en `ApiError(502, "SERVICIO_EXTERNO", …)`.
   - Revisar la cuota gratuita de Google Cloud antes de las pruebas en grupo.
2. **Extracción de parámetros (`comandos.py`)**: llenar `parametros` con lo que el texto permite deducir,
   usando los nombres de campo del `esquema_entrada` de cada modelo. Mínimo:
   - "mañana" → `dias_adelante: 1`, "pasado mañana" → 2, "en N días" → N (bitcoin, sp500).
   - Nombre de empresa o símbolo → `simbolo` (sp500).
3. **Decisión según la emoción**: si llega `emocion`, adaptar `respuesta_texto` con una tabla
   emoción → tono (p. ej. tristeza → empático, enojo → breve y calmado, felicidad → entusiasta). Debe ser
   determinista y estar probado. En el documento se describe como la regla de decisión del agente.
4. **Catálogo de comandos**: sale de `GET /api/modelos` (`comandos` de cada modelo); no duplicarlo aquí.

## Diseño del agente (para el documento)

Describir con PEAS (Russell & Norvig):
- **Rendimiento:** comandos correctamente interpretados y ejecutados; tiempo de respuesta.
- **Entorno:** usuario hablando frente al computador (ruido, acento costarricense).
- **Actuadores:** respuesta en pantalla y voz sintetizada; ejecución del modelo de ML.
- **Sensores:** micrófono (más la emoción que entrega la cámara).
- **Tipo de agente:** reflejo basado en modelo (reglas comando → acción, con el estado de la emoción).

## Criterios de aceptación

- [ ] Un audio real en español se transcribe (prueba de integración con un audio de muestra en `muestras/`,
      marcada con `skip` si no hay credenciales)
- [ ] "JarvisTEC precio del bitcoin para mañana" → `modelo: "bitcoin"`, `parametros: {"dias_adelante": 1}`
- [ ] La misma orden con `emocion: "tristeza"` y con `"felicidad"` produce respuestas distintas
- [x] Formato inválido → 415; audio vacío → 400; emoción desconocida → 422
- [ ] Las 4 pruebas anteriores están en `test_asistente.py`

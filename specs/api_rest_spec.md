# Especificación API REST — JarvisTEC

> **Fuente única de verdad (SDD).** El backend DEBE cumplir este contrato y el frontend DEBE consumirlo
> tal como está descrito. Cualquier cambio se acuerda entre ambos desarrolladores y se registra en la
> sección [Historial de cambios](#7-historial-de-cambios) **antes** de tocar código.

- **Base URL:** `http://127.0.0.1:8000` (app de escritorio PyWebView y `uvicorn` en desarrollo).
  En escritorio la interfaz se sirve desde el mismo origen, por lo que el frontend usa rutas relativas (`/api/...`).
- **Prefijo:** todos los endpoints viven bajo `/api`
- **Mocks:** los ejemplos JSON de este documento son los mocks oficiales del frontend
  (`frontend/src/api/mocks.js`). Si cambia un ejemplo aquí, se actualiza el mock en el mismo commit.
- **Formato:** JSON UTF-8 (excepto las subidas de archivos, que usan `multipart/form-data`)
- **Documentación interactiva:** `GET /docs` (Swagger) y `GET /openapi.json`, generadas por FastAPI
- **Idioma de campos:** español, `snake_case`

---

## 1. Convenciones generales

### 1.1 Formato de error

Toda respuesta con código HTTP 4xx/5xx tiene esta forma:

```json
{
  "error": {
    "codigo": "MODELO_NO_ENTRENADO",
    "mensaje": "El modelo 'bitcoin' aún no ha sido entrenado.",
    "detalle": null
  }
}
```

| HTTP | `codigo`               | Cuándo                                                        |
|------|------------------------|---------------------------------------------------------------|
| 400  | `SOLICITUD_INVALIDA`   | Parámetros con valores inválidos (p. ej. archivo vacío)       |
| 404  | `NO_ENCONTRADO`        | Ruta o modelo inexistente                                     |
| 415  | `TIPO_NO_SOPORTADO`    | Formato de audio/imagen no soportado                          |
| 422  | `VALIDACION`           | El cuerpo no cumple el esquema; `detalle` lista los campos    |
| 500  | `ERROR_INTERNO`        | Error no controlado                                           |
| 501  | `NO_IMPLEMENTADO`      | Endpoint definido en el contrato pero aún sin implementar     |
| 502  | `SERVICIO_EXTERNO`     | Falló Azure / Google Cloud                                    |
| 503  | `MODELO_NO_ENTRENADO`  | No existe el artefacto `modelo.joblib` del modelo             |

### 1.2 CORS

El backend permite el origen del servidor de desarrollo de Vite (`http://localhost:5173`).

---

## 2. Sistema

### `GET /api/salud`

Verifica que la API está arriba.

**200**
```json
{ "estado": "ok", "version": "0.1.0" }
```

### `GET /api/modelos`

Lista los modelos de ML registrados (se descubren automáticamente desde `backend/features/modelo_*`).

**200**
```json
{
  "modelos": [
    {
      "id": "modelo_01_bitcoin",
      "slug": "bitcoin",
      "nombre": "Predicción del precio del Bitcoin",
      "tipo": "regresion",
      "comandos": ["precio del bitcoin", "bitcoin mañana"],
      "entrenado": false
    }
  ]
}
```

`tipo` ∈ `"regresion" | "clasificacion" | "recomendacion"`.

---

## 3. Modelos de ML

Cada modelo expone el mismo par de endpoints bajo `/api/modelos/{slug}`. Los `slug` válidos están en
[`modelos_spec.md`](modelos_spec.md).

### `GET /api/modelos/{slug}/info`

Metadatos del modelo más las métricas de evaluación del último entrenamiento.

**200**
```json
{
  "id": "modelo_02_autos",
  "slug": "autos",
  "nombre": "Predicción del precio de un automóvil",
  "tipo": "regresion",
  "comandos": ["precio de un auto", "cuánto vale mi carro"],
  "entrenado": true,
  "metricas": { "r2": 0.95, "mae": 0.61, "rmse": 0.98 },
  "entrada_ejemplo": { "year": 2014, "present_price": 5.59, "...": "..." },
  "esquema_entrada": {
    "type": "object",
    "required": ["year", "present_price", "..."],
    "properties": {
      "year": { "type": "integer", "minimum": 1990, "maximum": 2030 },
      "fuel_type": { "type": "string", "enum": ["Petrol", "Diesel", "CNG"] }
    }
  }
}
```

- `metricas` es `null` si el modelo no está entrenado.
- `entrada_ejemplo` sirve para prellenar el formulario del modelo en la interfaz.
- `esquema_entrada` es el JSON Schema del cuerpo de `/predecir` (generado desde la clase `Entrada` del
  `router.py`). La interfaz construye el formulario de cada modelo a partir de él, sin código propio por modelo.

### `POST /api/modelos/{slug}/predecir`

**Cuerpo:** objeto JSON plano con las características de entrada del modelo. El esquema exacto de cada
modelo se documenta en [`modelos_spec.md`](modelos_spec.md) y en `/docs`.

**200**
```json
{
  "modelo": "autos",
  "prediccion": 3.42,
  "unidad": "lakhs INR",
  "probabilidades": null,
  "texto": "El precio estimado del vehículo es 3.42 lakhs INR."
}
```

| Campo            | Tipo                       | Notas                                                         |
|------------------|----------------------------|---------------------------------------------------------------|
| `modelo`         | string                     | `slug` del modelo                                             |
| `prediccion`     | number \| string \| array  | Valor numérico (regresión), clase (clasificación) o lista (recomendación) |
| `unidad`         | string \| null             | Unidad del valor predicho, si aplica                          |
| `probabilidades` | object \| null             | Solo clasificación: `{ "clase": probabilidad }`               |
| `texto`          | string                     | Frase lista para que JARVIS la lea en voz alta                |

**Errores:** 422 (entrada inválida), 503 (modelo no entrenado).

---

## 4. Asistente de voz (`features/asistente_voz`)

### `POST /api/voz/transcribir`

Convierte audio a texto con Google Cloud Speech-to-Text (o Azure Speech).

**Cuerpo:** `multipart/form-data`

| Campo    | Tipo   | Requerido | Notas                                   |
|----------|--------|-----------|-----------------------------------------|
| `audio`  | file   | sí        | `audio/webm` (MediaRecorder) o `audio/wav` |
| `idioma` | string | no        | BCP-47, por defecto `es-CR`             |

**200**
```json
{ "texto": "jarvis precio del bitcoin para mañana", "confianza": 0.93, "idioma": "es-CR" }
```

**Errores:** 400 (audio vacío), 415 (formato), 502 (falla del proveedor).

### `POST /api/asistente/comando`

Interpreta un texto (transcrito o escrito) y lo asocia a un modelo de ML.

**Cuerpo**
```json
{ "texto": "JarvisTEC precio del bitcoin para mañana", "emocion": "tristeza" }
```

| Campo     | Tipo           | Requerido | Notas                                                                 |
|-----------|----------------|-----------|-----------------------------------------------------------------------|
| `texto`   | string         | sí        | Texto transcrito o escrito                                            |
| `emocion` | string \| null | no        | Última `emocion_dominante` detectada por §5 (una de sus 8 claves)     |

**200**
```json
{
  "reconocido": true,
  "modelo": "bitcoin",
  "parametros": { "dias_adelante": 1 },
  "emocion": "tristeza",
  "respuesta_texto": "Te noto algo decaído; vamos a ver si el bitcoin te da una buena noticia. Consultando el modelo de predicción del precio del Bitcoin."
}
```

- Si `reconocido` es `false`, `modelo` es `null` y `respuesta_texto` explica que no se entendió el comando.
- **Decisión según la emoción** (enunciado: "el sistema interpretará el rostro y la voz del usuario y
  tomará una decisión"): si llega `emocion`, `respuesta_texto` adapta el tono a esa emoción y `emocion`
  la devuelve tal cual. Si no llega, `emocion` es `null` y la respuesta es neutral.
- El frontend usa `modelo` + `parametros` para llamar a `POST /api/modelos/{slug}/predecir`.
  Si `parametros` completa el `esquema_entrada` (o el modelo no requiere entrada, como las series de
  tiempo), la predicción se **ejecuta de inmediato**; si no, se abre el formulario del modelo prellenado.
- La asociación texto → modelo usa la lista `comandos` de cada modelo (ver `GET /api/modelos`).

---

## 5. Visión facial (`features/vision_facial`)

### `POST /api/vision/emocion`

Detecta rostros en un frame de la cámara y estima la emoción de cada uno.

**Cuerpo:** `multipart/form-data`

| Campo    | Tipo | Requerido | Notas                    |
|----------|------|-----------|--------------------------|
| `imagen` | file | sí        | `image/jpeg` o `image/png` |

**200**
```json
{
  "cantidad": 1,
  "rostros": [
    {
      "rectangulo": { "x": 120, "y": 80, "ancho": 160, "alto": 160 },
      "emocion_dominante": "felicidad",
      "puntajes": {
        "felicidad": 0.91, "tristeza": 0.01, "enojo": 0.0, "sorpresa": 0.05,
        "miedo": 0.0, "desprecio": 0.0, "disgusto": 0.0, "neutral": 0.03
      }
    }
  ]
}
```

- Las claves de `puntajes` son fijas (las 8 de arriba); si el proveedor no entrega alguna, su valor es `0.0`.
- Si no hay rostros: `cantidad = 0`, `rostros = []` (no es error).

**Errores:** 400 (imagen vacía), 415 (formato), 502 (falla del proveedor).

---

## 6. Variables de entorno (`backend/.env`)

| Variable                          | Uso                                         |
|-----------------------------------|---------------------------------------------|
| `AZURE_FACE_ENDPOINT`             | Endpoint del recurso Azure AI Face          |
| `AZURE_FACE_KEY`                  | Clave del recurso Azure AI Face             |
| `GOOGLE_APPLICATION_CREDENTIALS`  | Ruta al JSON de la cuenta de servicio de GCP |
| `CORS_ORIGENES`                   | Orígenes permitidos, separados por coma     |

Ver `backend/.env.example`. El archivo `.env` **nunca** se sube a git.

---

## 7. Historial de cambios

| Fecha      | Autor | Cambio                         |
|------------|-------|--------------------------------|
| 2026-10-03 | —     | Versión inicial del contrato   |
| 2026-10-03 | —     | Modo escritorio (PyWebView, mismo origen) y regla de mocks |
| 2026-10-03 | —     | `esquema_entrada` en `/info`; `emocion` en `/api/asistente/comando` (decisión según la emoción) |

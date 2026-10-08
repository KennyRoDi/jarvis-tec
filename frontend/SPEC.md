# SPEC — Interfaz JarvisTEC

> **Responsable:** Dev B · **Contrato:** [`specs/api_rest_spec.md`](../specs/api_rest_spec.md) (consumido solo
> a través de `src/api/cliente.js`) · **Trazabilidad:** [`specs/alcance_spec.md`](../specs/alcance_spec.md) (R1, A1, A2, rubro Aplicación)

## Objetivo

Plataforma para comunicar al usuario con la máquina, con la estética de la Figura 1 del enunciado (HUD
sci-fi oscuro con acentos cian). Corre dentro de la ventana PyWebView. Cada uno de los 10 modelos debe poder
usarse desde aquí: **vale 1 punto por modelo** (rubro Aplicación, 10 %).

## Zonas de la pantalla

| Zona                | Contenido                                                                              |
|---------------------|----------------------------------------------------------------------------------------|
| Núcleo animado      | El "rostro" de JARVIS (el enunciado dice que no tiene rostro propio). Estados: reposo, escuchando, pensando, hablando |
| Cámara              | Video en vivo, recuadro sobre el rostro y emoción dominante con sus puntajes            |
| Barra de comando    | Botón para hablar y campo de texto como alternativa                                    |
| Panel de modelos    | Las 10 tarjetas de `GET /api/modelos`, con estado (entrenado o no) y métricas de `/info` |
| Formulario de modelo | Generado desde `esquema_entrada` y prellenado con `entrada_ejemplo`; ningún formulario escrito a mano |
| Respuesta           | `texto` de la predicción, `probabilidades` (si es clasificación) e historial de la sesión |

## Flujos

1. **Emoción (A1):** `getUserMedia({video})` → cada 3–5 s un frame a JPEG (`canvas.toBlob`) →
   `detectarEmocion` → guardar la última `emocion_dominante`.
2. **Voz (A2):** mantener presionado para hablar → `MediaRecorder` (`audio/webm`) → `transcribir` →
   `interpretarComando(texto, emocionActual)` → si `parametros` completa el `esquema_entrada`, `predecir`
   de inmediato; si no, abrir el formulario prellenado → mostrar el resultado y **leerlo en voz alta**.
3. **Respuesta hablada:** `window.speechSynthesis` (`es-*`), con el núcleo en estado "hablando" mientras
   suena (la "animación de audio" del enunciado). Verificar que funcione dentro de PyWebView; si no, mostrar
   el texto con la animación igualmente.
4. **Errores:** mostrar `ApiError.mensaje`. 503 → "modelo aún no entrenado"; 501 → "función en desarrollo".

## Reglas

- Todas las llamadas pasan por `src/api/cliente.js`; rutas relativas `/api/...`, sin URLs fijas.
- Para trabajar sin backend: `VITE_USAR_MOCKS=true npm run dev`. Si cambia la spec, se actualiza `src/api/mocks.js`.
- Para la app de escritorio: `npm run build`. FastAPI sirve `frontend/dist` automáticamente.
- Cámara y micrófono: en Linux los concede `backend/core/escritorio.py` (verificado). **En Windows (WebView2)
  está sin verificar**: probar al inicio y registrar el resultado aquí.

## Criterios de aceptación

- [ ] Estética HUD inspirada en la Figura 1, legible a 1280×800 (tamaño de la ventana)
- [ ] La cámara muestra video y la emoción se actualiza sola
- [ ] Un comando de voz completo produce una respuesta mostrada y hablada
- [ ] Los 10 modelos se pueden ejecutar desde su formulario (1 punto cada uno)
- [ ] Funciona con mocks y con el backend real
- [ ] Cámara y micrófono verificados en Linux y en Windows dentro de PyWebView

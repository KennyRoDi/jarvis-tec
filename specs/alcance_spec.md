# Alcance — trazabilidad del enunciado

Cada requisito del enunciado (Proyecto en Grupos, IA I Semestre 2026, Ing. Efrén Jiménez Delgado) se
asocia aquí con la parte del proyecto que lo cumple y con el `SPEC.md` que guía su desarrollo.
**Antes de cerrar la entrega, todas las filas deben estar en ✅.**

Leyenda: ✅ cumplido · 🟡 estructura lista, falta implementar · ⏳ pendiente · ❓ requiere decisión del profesor

## Aspectos administrativos

| Requisito                                              | Cómo se cumple                                   | Estado |
|--------------------------------------------------------|--------------------------------------------------|--------|
| Máximo 3 integrantes                                   | 2 desarrolladores                                | ✅ |
| Entrega en semana 11 (la tabla de entregas dice **semana 10**) | Confirmar la fecha con el profesor       | ❓ |
| Documentación en Overleaf sin excepción                | `docs_latex/` → [SPEC](../docs_latex/SPEC.md)    | 🟡 |
| Repositorio en Bitbucket, GitLab o GitHub con todo el proyecto | GitHub `KennyRoDi/jarvis-tec`            | ✅ |

## Descripción general y requerimientos

| Requisito                                                                 | Cómo se cumple                                                        | Estado |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------|--------|
| Asistente personal que responde preguntas sobre los problemas de ML       | Comandos → modelos (`/api/asistente/comando` + `/predecir`)           | 🟡 |
| "El sistema interpretará el rostro y la voz del usuario y **tomará una decisión** con respecto a eso" | La emoción detectada viaja en `/api/asistente/comando` y modifica la respuesta → [asistente_voz](../backend/features/asistente_voz/SPEC.md) | ⏳ |
| Sin rostro propio: animación o foto ilustrativa + animación de audio al responder | Núcleo animado + respuesta hablada (TTS) → [frontend](../frontend/SPEC.md) | ⏳ |
| R1. Plataforma de software para comunicar usuario y máquina (Figura 1)    | App de escritorio PyWebView + interfaz React estilo Jarvis            | 🟡 |
| R2. API REST con los endpoints necesarios                                 | FastAPI, contrato en [api_rest_spec.md](api_rest_spec.md)             | 🟡 |
| R3. 10 modelos de la lista proporcionada                                  | [modelos_spec.md](modelos_spec.md) (todos de la lista: no requieren aprobación de dataset) | 🟡 1/10 |
| R4. Alternativa con dataset externo (opcional)                            | No se usa                                                             | — |

## Alcances (entregables de la semana 10)

| Alcance                                                                    | Cómo se cumple                                                                 | Estado |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------|
| A1. Agente inteligente que reconoce el sentimiento en una foto de cámara en tiempo real (frame de video) | Cámara en la interfaz → `POST /api/vision/emocion` (Azure Face + Google Vision) → [vision_facial](../backend/features/vision_facial/SPEC.md) | ⏳ |
| A2. Agente inteligente que convierte audio a texto y **ejecuta** la instrucción | Micrófono → `POST /api/voz/transcribir` (Google) → `/api/asistente/comando` → `/predecir` | ⏳ |
| A3. Al menos 10 algoritmos de aprendizaje automático de la lista           | `backend/features/modelo_01 … modelo_10`, cada uno con su `SPEC.md`            | 🟡 1/10 |
| A4. Conjunto de comandos asociados a los modelos (ej. JarvisTEC "tipo de cambio para mañana") | `MODELO_INFO["comandos"]` de cada modelo + catálogo en los anexos del documento | 🟡 |
| A1 y A2 piden **diseñar un modelo de agente**                              | Diseño PEAS y tipo de agente en el documento → [docs_latex](../docs_latex/SPEC.md) | ⏳ |
| Cámara y micrófono dentro de la app de escritorio                          | `core/escritorio.py` concede `getUserMedia` en Linux (verificado con webcam y micrófono reales). Windows sin verificar | 🟡 |

## Notas para la primera entrega

| Nota                                                       | Cómo se cumple                                   | Estado |
|------------------------------------------------------------|--------------------------------------------------|--------|
| Usar Azure para reconocer rostros                          | Azure Face detecta el rostro; Google Vision da la emoción (opción 1) | 🟡 decidido el 2026-10-08, falta informar al profesor; ver [SPEC](../backend/features/vision_facial/SPEC.md) |
| Usar Speech-to-Text para pasar audio a texto               | `asistente_voz` con Google Cloud Speech-to-Text  | ⏳ |
| Solo en esta entrega se permiten APIs de Google/Azure para sentimientos y voz | El contrato no depende del proveedor: en una etapa posterior se reemplaza la implementación sin tocar el frontend | ✅ diseño |

## Documentación (componentes obligatorios)

Portada · Solución planteada · Arquitecturas de ML · Justificación y explicación del modelo (basada en
**artículos científicos**) · Análisis de resultados · Bibliografía · Anexos → [docs_latex/SPEC.md](../docs_latex/SPEC.md). Estado: 🟡 esqueleto.

## Evaluación (cómo se gana cada punto)

| Rubro                 | Peso | Condición para el punto                                                  | Dónde se verifica |
|-----------------------|------|--------------------------------------------------------------------------|-------------------|
| Creación del modelo   | 30 % | Por modelo: análisis 0.5 · entendimiento 0.5 · exploración 0.5 · modelo 2 · evaluación 1 · conclusión 0.5 | `train.py` + `analisis.md` de cada modelo |
| Aplicación            | 10 % | 1 punto por modelo **usable desde la interfaz**                           | Formulario del modelo en el frontend |
| API REST              | 10 % | 1 punto por modelo **expuesto y funcionando** en el API                    | `POST /api/modelos/{slug}/predecir` + prueba |
| Documento             | 10 % | Completo 10 · incompleto 3 · sin presentar 0                             | `docs_latex/` en Overleaf |

Por lo tanto, un modelo solo está **terminado** cuando cumple las tres cosas: entrenado y documentado, con endpoint y prueba, y usable desde la interfaz. Los criterios están en el `SPEC.md` de cada modelo.

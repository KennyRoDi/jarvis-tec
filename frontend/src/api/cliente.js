// Cliente de la API de JarvisTEC. Cada función corresponde a un endpoint de specs/api_rest_spec.md.
// Con VITE_USAR_MOCKS=true responde con los mocks del contrato (mocks.js) sin llamar al backend.

import { mocks } from './mocks'

export class ApiError extends Error {
  constructor(status, { codigo, mensaje, detalle }) {
    super(mensaje)
    this.status = status
    this.codigo = codigo
    this.detalle = detalle
  }
}

async function solicitar(ruta, opciones = {}) {
  const respuesta = await fetch(ruta, opciones)
  const cuerpo = await respuesta.json().catch(() => null)
  if (!respuesta.ok) {
    throw new ApiError(respuesta.status, cuerpo?.error ?? { codigo: 'ERROR_INTERNO', mensaje: respuesta.statusText })
  }
  return cuerpo
}

const json = (cuerpo) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(cuerpo),
})

const archivo = (campo, blob, nombre, extras = {}) => {
  const datos = new FormData()
  datos.append(campo, blob, nombre)
  Object.entries(extras).forEach(([clave, valor]) => datos.append(clave, valor))
  return { method: 'POST', body: datos }
}

const real = {
  // §2 Sistema
  salud: () => solicitar('/api/salud'),
  listarModelos: () => solicitar('/api/modelos'),

  // §3 Modelos de ML
  infoModelo: (slug) => solicitar(`/api/modelos/${slug}/info`),
  predecir: (slug, entrada) => solicitar(`/api/modelos/${slug}/predecir`, json(entrada)),

  // §4 Asistente de voz
  transcribir: (audioBlob, idioma = 'es-CR') =>
    solicitar('/api/voz/transcribir', archivo('audio', audioBlob, 'audio.webm', { idioma })),
  interpretarComando: (texto, emocion = null) => solicitar('/api/asistente/comando', json({ texto, emocion })),

  // §5 Visión facial
  detectarEmocion: (imagenBlob) => solicitar('/api/vision/emocion', archivo('imagen', imagenBlob, 'frame.jpg')),
}

const api = import.meta.env.VITE_USAR_MOCKS === 'true' ? mocks : real

export const {
  salud, listarModelos, infoModelo, predecir, transcribir, interpretarComando, detectarEmocion,
} = api

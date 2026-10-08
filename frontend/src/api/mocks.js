// Mocks del contrato (copiados de los ejemplos de specs/api_rest_spec.md).
// Se activan con VITE_USAR_MOCKS=true para trabajar la interfaz sin el backend.

const esperar = (ms = 400) => new Promise((resolver) => setTimeout(resolver, ms))

export const mocks = {
  salud: async () => ({ estado: 'ok', version: '0.1.0' }),

  listarModelos: async () => ({
    modelos: [
      {
        id: 'modelo_01_bitcoin', slug: 'bitcoin', nombre: 'Predicción del precio del Bitcoin',
        tipo: 'regresion', comandos: ['precio del bitcoin', 'bitcoin mañana'], entrenado: false,
      },
      {
        id: 'modelo_02_autos', slug: 'autos', nombre: 'Predicción del precio de un automóvil',
        tipo: 'regresion', comandos: ['precio de un auto', 'cuánto vale mi carro'], entrenado: true,
      },
    ],
  }),

  infoModelo: async (slug) => ({
    id: `modelo_xx_${slug}`, slug, nombre: `Modelo ${slug}`, tipo: 'regresion', comandos: [],
    entrenado: true, metricas: { r2: 0.95, mae: 0.61, rmse: 0.98 },
    entrada_ejemplo: { year: 2014, fuel_type: 'Petrol' },
    esquema_entrada: {
      type: 'object',
      required: ['year', 'fuel_type'],
      properties: {
        year: { type: 'integer', minimum: 1990, maximum: 2030 },
        fuel_type: { type: 'string', enum: ['Petrol', 'Diesel', 'CNG'] },
      },
    },
  }),

  predecir: async (slug) => {
    await esperar()
    return {
      modelo: slug, prediccion: 3.42, unidad: 'lakhs INR', probabilidades: null,
      texto: 'El precio estimado del vehículo es 3.42 lakhs INR.',
    }
  },

  transcribir: async () => {
    await esperar(800)
    return { texto: 'jarvis precio del bitcoin para mañana', confianza: 0.93, idioma: 'es-CR' }
  },

  interpretarComando: async (texto, emocion = null) => ({
    reconocido: true, modelo: 'bitcoin', parametros: { dias_adelante: 1 }, emocion,
    respuesta_texto: 'Consultando el modelo de predicción del precio del Bitcoin.',
  }),

  detectarEmocion: async () => {
    await esperar()
    return {
      cantidad: 1,
      rostros: [{
        rectangulo: { x: 120, y: 80, ancho: 160, alto: 160 },
        emocion_dominante: 'felicidad',
        puntajes: {
          felicidad: 0.91, tristeza: 0.01, enojo: 0.0, sorpresa: 0.05,
          miedo: 0.0, desprecio: 0.0, disgusto: 0.0, neutral: 0.03,
        },
      }],
    }
  },
}

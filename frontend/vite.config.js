import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Las llamadas a /api van al backend FastAPI (specs/api_rest_spec.md)
    proxy: { '/api': 'http://localhost:8000' },
  },
})

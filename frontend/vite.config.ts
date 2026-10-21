import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// In development the FastAPI backend runs on :8000; in production it serves this build itself.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})

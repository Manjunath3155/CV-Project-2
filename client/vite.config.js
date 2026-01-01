import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['.ngrok-free.app'], // Allow your ngrok tunnel
    host: true // Expose to network (required for ngrok to forward)
  }
})

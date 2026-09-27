import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Localhost only by default. Set to true to open the app to the LAN
    // (see README "같은 LAN의 다른 PC에서 접속").
    host: false,
  },
})

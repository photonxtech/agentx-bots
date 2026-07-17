import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('@mui') || id.includes('@emotion')) return 'mui'
          if (id.includes('framer-motion')) return 'motion'
          if (id.includes('react-markdown')) return 'markdown'
          if (id.includes('@tanstack/react-query') || id.includes('axios')) return 'query'
        },
      },
    },
  },
})

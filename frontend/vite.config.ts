import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/profile': 'http://localhost:8000',
      '/insights': 'http://localhost:8000',
      '/concepts': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});

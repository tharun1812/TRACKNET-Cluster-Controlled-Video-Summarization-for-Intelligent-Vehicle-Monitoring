import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// During development the dashboard proxies /api to the Flask server (python -m tracknet serve).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:5000' },
  },
});

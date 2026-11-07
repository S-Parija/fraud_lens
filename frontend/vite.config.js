import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// React calls /api on its own origin; Vite forwards those calls to FastAPI.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});

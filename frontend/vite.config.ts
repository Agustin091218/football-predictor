import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/predict": "http://localhost:8000",
      "/leagues": "http://localhost:8000",
      "/world-cup": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/predictions": "http://localhost:8000",
      "/dashboard": "http://localhost:8000",
    },
  },
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API and websocket traffic to the python viewer server
// (`python -m tinker_cookbook.tokendb_studio.serve [log_path=...] port=7423`).
// `npm run build` emits the static bundle the python server ships as
// package data (tinker_cookbook/tokendb_studio/static/).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7423",
        ws: true, // per-run websockets live under /api/runs/{run_id}/ws
      },
      "/ws": { target: "ws://127.0.0.1:7423", ws: true },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
});

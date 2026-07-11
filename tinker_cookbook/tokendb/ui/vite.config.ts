import { defineConfig } from "vite";

// Dev server proxies API and websocket traffic to the python viewer server
// (`python -m tinker_cookbook.tokendb.serve log_path=... port=7423`).
// `npm run build` emits the static bundle the python server ships as
// package data (tinker_cookbook/tokendb/static/).
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://localhost:7423",
      "/ws": { target: "ws://localhost:7423", ws: true },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
});

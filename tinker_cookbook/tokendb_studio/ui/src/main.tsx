import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { detectMode } from "./api";
import { App } from "./App";
import "./style.css";

// Probe the server mode once, then mount: routing depends on whether the
// server is showing one run (single-run mode) or the whole registry.
void detectMode().then((mode) => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App mode={mode} />
    </StrictMode>,
  );
});

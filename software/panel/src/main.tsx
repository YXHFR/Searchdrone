import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme/tokens.css";
import "./theme/app.css";
import "./i18n";
import { App } from "./app/App";
import { installTestHooks } from "./lib/testHooks";

installTestHooks();
const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme/tokens.css";
import "./theme/base.css";
import "./theme/shell.css";
import "./theme/safety.css";
import "./theme/chat.css";
import "./theme/map.css";
import "./theme/video.css";
import "./theme/views.css";
import "./theme/candidates.css";
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

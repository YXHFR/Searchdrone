import { create } from "zustand";

export type Theme = "dark" | "light";
export interface LogEntry {
  t: number;
  level: "info" | "critical";
  message: string;
}

const THEME_KEY = "sd.panel.theme";
const MAX_LOG = 200;

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // ignore
  }
  if (typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

interface UiState {
  theme: Theme;
  eventLog: LogEntry[];
  /** Latest critical message, read out with aria-live="assertive". */
  alert: string | null;
  setTheme(t: Theme): void;
  log(level: LogEntry["level"], message: string): void;
}

export const useUiStore = create<UiState>((set) => ({
  theme: initialTheme(),
  eventLog: [],
  alert: null,
  setTheme: (theme) => {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // ignore
    }
    set({ theme });
  },
  log: (level, message) =>
    set((s) => ({
      eventLog: [{ t: Date.now(), level, message }, ...s.eventLog].slice(0, MAX_LOG),
      alert: level === "critical" ? message : s.alert,
    })),
}));

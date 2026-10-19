import { useSyncExternalStore } from "react";
import { useUiStore } from "../state/uiStore";

/** What the operator chose; "auto" follows the system setting. */
export type ThemePref = "auto" | "dark" | "light";
export type Theme = "dark" | "light";

export const THEME_PREFS: readonly ThemePref[] = ["auto", "dark", "light"];

export function resolveTheme(pref: ThemePref, systemLight: boolean): Theme {
  if (pref === "auto") return systemLight ? "light" : "dark";
  return pref;
}

const QUERY = "(prefers-color-scheme: light)";

function subscribe(onChange: () => void): () => void {
  if (typeof matchMedia !== "function") return () => {};
  const mq = matchMedia(QUERY);
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}

const systemLight = (): boolean => typeof matchMedia === "function" && matchMedia(QUERY).matches;

/** The theme in force: the operator's choice, or the system's when that is "auto". */
export function useTheme(): Theme {
  const pref = useUiStore((s) => s.themePref);
  const light = useSyncExternalStore(subscribe, systemLight, () => false);
  return resolveTheme(pref, light);
}

import { create } from "zustand";
import type { ThemePref } from "../theme/theme";

export interface LogEntry {
  t: number;
  level: "info" | "critical";
  message: string;
}

const THEME_KEY = "sd.panel.theme";
const MAX_LOG = 200;

function initialThemePref(): ThemePref {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "auto" || saved === "dark" || saved === "light") return saved;
  } catch {
    // ignore
  }
  return "auto";
}

export interface Toast {
  id: number;
  text: string;
  /** Approval card to show when the toast's "Visa" is pressed. */
  approvalId?: string | undefined;
  /** An alarm: stays until "Jag har sett det" and is read out at once (role="alert"). */
  sticky?: boolean;
}

/** The chat composer's text and reply target, so "Ändra" and "Svara" can fill and focus it. */
export interface Draft {
  text: string;
  replyTo: { id: string; preview: string } | null;
  /** Bumped to move focus to the composer. */
  focusTick: number;
}

interface UiState {
  themePref: ThemePref;
  eventLog: LogEntry[];
  /** Latest critical message, read out with aria-live="assertive". */
  alert: string | null;
  toasts: Toast[];
  draft: Draft;
  setDraft(patch: Partial<Omit<Draft, "focusTick">>, focus?: boolean): void;
  /** Approval card that should be brought into view and focused (set by "Visa"). */
  focusApproval: string | null;
  /** Latest new approval, announced with aria-live="assertive". */
  approvalAnnounce: string | null;
  setThemePref(p: ThemePref): void;
  log(level: LogEntry["level"], message: string): void;
  toast(text: string, approvalId?: string): void;
  /** A critical alert that stays until acknowledged; also goes in the event log. */
  alarm(text: string): void;
  dismissToast(id: number): void;
  approvalRequested(approvalId: string): void;
  showApproval(approvalId: string | null): void;
}

const TOAST_MS = 8_000;
const MAX_TOASTS = 4;
let toastSeq = 0;

/** At most MAX_TOASTS passing notices, oldest out first; an alarm is never pushed out. */
function trimToasts(list: Toast[]): Toast[] {
  let extra = list.filter((x) => !x.sticky).length - MAX_TOASTS;
  return list.filter((x) => x.sticky || extra-- <= 0);
}

export const useUiStore = create<UiState>((set, get) => ({
  themePref: initialThemePref(),
  eventLog: [],
  alert: null,
  toasts: [],
  draft: { text: "", replyTo: null, focusTick: 0 },
  setDraft: (patch, focus = false) =>
    set((s) => ({
      draft: { ...s.draft, ...patch, focusTick: s.draft.focusTick + (focus ? 1 : 0) },
    })),
  focusApproval: null,
  approvalAnnounce: null,
  setThemePref: (themePref) => {
    try {
      localStorage.setItem(THEME_KEY, themePref);
    } catch {
      // ignore
    }
    set({ themePref });
  },
  log: (level, message) =>
    set((s) => ({
      eventLog: [{ t: Date.now(), level, message }, ...s.eventLog].slice(0, MAX_LOG),
      alert: level === "critical" ? message : s.alert,
    })),
  toast: (text, approvalId) => {
    toastSeq += 1;
    const id = toastSeq;
    set((s) => ({ toasts: trimToasts([...s.toasts, { id, text, approvalId }]) }));
    setTimeout(() => get().dismissToast(id), TOAST_MS);
  },
  alarm: (text) => {
    toastSeq += 1;
    const id = toastSeq;
    // The toast's role="alert" reads it out, so the log entry leaves `alert` alone.
    set((s) => ({
      eventLog: [{ t: Date.now(), level: "critical" as const, message: text }, ...s.eventLog].slice(
        0,
        MAX_LOG,
      ),
      toasts: trimToasts([...s.toasts, { id, text, sticky: true }]),
    }));
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  approvalRequested: (approvalId) => {
    set({ approvalAnnounce: approvalId });
    get().toast("", approvalId);
  },
  showApproval: (approvalId) => set({ focusApproval: approvalId }),
}));

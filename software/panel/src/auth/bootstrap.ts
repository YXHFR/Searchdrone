import { api, HttpError, onUnauthorized } from "../api/http";
import { paths } from "../api/paths";
import type { PanelConfig, Session } from "../api/types";
import { rememberVehicleId, useSessionStore } from "../state/sessionStore";

export const BOOT_TIMING = { emergencyAfterMs: 3_000, retryEveryMs: 2_000, reauthPollMs: 5_000 };

/** Exchange a one-time login code from `#login=<code>`, then drop it from the address. */
async function consumeLoginFragment(): Promise<void> {
  const m = /(?:^#|&)login=([^&]+)/.exec(window.location.hash);
  if (!m?.[1]) return;
  const code = decodeURIComponent(m[1]);
  history.replaceState(null, "", window.location.pathname + window.location.search);
  try {
    await api<Session>(paths.session(), {
      method: "POST",
      body: { login_code: code },
      quiet401: true,
    });
  } catch {
    // a wrong or used code just ends on the login page
  }
}

async function loadSessionAndConfig(): Promise<"ready" | "login"> {
  const store = useSessionStore.getState();
  let session: Session;
  try {
    session = await api<Session>(paths.session(), {
      quiet401: true,
      timeoutMs: BOOT_TIMING.emergencyAfterMs,
    });
  } catch (e) {
    if (e instanceof HttpError && e.status === 401) {
      store.set({ status: "login", session: null });
      return "login";
    }
    throw e;
  }
  const config = await api<PanelConfig>(paths.panelConfig(), {
    timeoutMs: BOOT_TIMING.emergencyAfterMs,
  });
  rememberVehicleId(config.default_vehicle_id);
  store.set({ status: "ready", session, config, needsReauth: false, is_remote: false });
  return "ready";
}

let running: Promise<void> | null = null;

/**
 * Start-up: safety bar first (already rendered), then login fragment, session and config.
 * No answer within 3 s (other than 401) → emergency page, retried every 2 s.
 */
export function boot(): Promise<void> {
  if (running) return running;
  running = (async () => {
    await consumeLoginFragment();
    const emergency = setTimeout(() => {
      if (useSessionStore.getState().status === "loading")
        useSessionStore.getState().set({ status: "emergency" });
    }, BOOT_TIMING.emergencyAfterMs);
    for (;;) {
      try {
        await loadSessionAndConfig();
        break;
      } catch {
        useSessionStore.getState().set({ status: "emergency" });
        await new Promise((r) => setTimeout(r, BOOT_TIMING.retryEveryMs));
      }
    }
    clearTimeout(emergency);
  })().finally(() => {
    running = null;
  });
  return running;
}

/** PIN login from the login page or the re-login sheet. Returns an i18n error key or null. */
export async function loginWithPin(pin: string): Promise<string | null> {
  try {
    await api<Session>(paths.session(), { method: "POST", body: { pin }, quiet401: true });
  } catch (e) {
    if (e instanceof HttpError) {
      if (e.status === 401 && e.code === "PIN_INVALIDATED") return "login.error.invalidated";
      if (e.status === 429) return "login.error.tooMany";
      if (e.status === 401) return "login.error.wrong";
    }
    return "login.error.noContact";
  }
  await boot();
  return null;
}

export async function logout(): Promise<void> {
  try {
    await api<void>(paths.logout(), { method: "POST", quiet401: true });
  } finally {
    useSessionStore.getState().set({ status: "login", session: null, needsReauth: false });
  }
}

/** Any 401 while the shell runs opens the re-login sheet; the page is never reloaded. */
export function watchUnauthorized(): () => void {
  return onUnauthorized(() => {
    const s = useSessionStore.getState();
    if (s.status === "ready") s.set({ needsReauth: true });
  });
}

/** While the re-login sheet is open: a login in another tab gives the same cookie. */
export async function pollSession(): Promise<boolean> {
  try {
    const session = await api<Session>(paths.session(), { quiet401: true });
    useSessionStore.getState().set({ session, needsReauth: false });
    return true;
  } catch {
    return false;
  }
}

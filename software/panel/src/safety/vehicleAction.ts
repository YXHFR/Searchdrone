// State machine behind STOP, RETURN HOME and Resume. Plain fetch on purpose: no cache,
// no library retries and no offline pause may stand between a press and the request.
//
// "acked" is reached ONLY through an ActionOutcome with the same action_id and
// result acked/already_active. A 202 means "sent", nothing more.
import { MUTATION_HEADERS, readError } from "../api/http";
import { paths, type VehicleAction } from "../api/paths";
import type { ActionDispatched, ActionDispatchedAll, ActionOutcome, ApiError } from "../api/types";
import { uuidv7 } from "../lib/uuidv7";

export type ActionState =
  | "idle"
  | "sending"
  | "retrying"
  | "sent"
  | "acked"
  | "unconfirmed"
  | "failed";

/** i18n keys for the text under the button; `detail` carries server text (detail_sv). */
export type ActionText =
  | { key: "safety.state.sending" }
  | { key: "safety.state.retrying" }
  | { key: "safety.state.sent" }
  | { key: "safety.state.acked" }
  | { key: "safety.state.unconfirmed" }
  | { key: "safety.state.failed" }
  | { key: "safety.error.forbidden" }
  | { key: "safety.error.noPath" }
  | { key: "safety.error.session" }
  | { key: "safety.error.noGround" }
  | { key: "safety.error.vehicleFailed" }
  | { key: "detail"; detail: string };

export interface ActionSnapshot {
  state: ActionState;
  text: ActionText | null;
  onboardReceived: boolean;
  clientActionId: string | null;
}

export type ActionLog = (level: "info" | "critical", message: string) => void;

export const TIMING = {
  attemptTimeoutMs: 3_000,
  retryDelayMs: 500,
  giveUpMs: 10_000,
  ackWaitMs: 3_000,
  pollAfterMs: 1_000,
  pollEveryMs: 500,
  pollForMs: 30_000,
  ackedShownMs: 4_000,
  alarmAfterMs: 30_000,
} as const;

const FINAL_OK = new Set(["acked", "already_active"]);

export interface MachineDeps {
  fetch: typeof fetch;
  log: ActionLog;
  onUnauthorized?: () => void;
  now?: () => number;
}

type Timer = ReturnType<typeof setTimeout>;

/**
 * One long-lived machine per button. The target vehicle is read at each press and frozen for
 * that press, so a vehicle id that arrives later (config loaded) never cancels a press in
 * flight. A null target uses the *_all endpoints.
 */
export class VehicleActionMachine {
  private snap: ActionSnapshot = {
    state: "idle",
    text: null,
    onboardReceived: false,
    clientActionId: null,
  };
  private listeners = new Set<(s: ActionSnapshot) => void>();
  private generation = 0;
  private pressedAt = 0;
  private timers = new Set<Timer>();
  /** action_id → vehicle_id for the current press; value true once that vehicle acked. */
  private pending = new Map<string, { vehicleId: string; acked: boolean }>();
  private readonly now: () => number;
  /** Target of the current press (null = all vehicles). */
  private target: string | null = null;

  constructor(
    private readonly vehicleId: () => string | null,
    private readonly action: VehicleAction,
    private readonly deps: MachineDeps,
  ) {
    this.now = deps.now ?? (() => Date.now());
  }

  get snapshot(): ActionSnapshot {
    return this.snap;
  }

  subscribe(fn: (s: ActionSnapshot) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** New press: always sends, with a new client_action_id, even mid-attempt. */
  trigger(): void {
    if (this.action === "resume" && this.vehicleId() === null) return; // Resume needs a vehicle
    this.clearTimers();
    this.generation += 1;
    this.pending.clear();
    this.pressedAt = this.now();
    this.target = this.vehicleId();
    const id = uuidv7();
    this.set({
      state: "sending",
      text: { key: "safety.state.sending" },
      onboardReceived: false,
      clientActionId: id,
    });
    this.deps.log("info", `${this.action}: tryck (${id})`);
    void this.attempt(this.generation, id);
  }

  /** ✕ on unconfirmed/failed. Closing the message is logged: the action was never confirmed. */
  dismiss(): void {
    if (this.snap.state !== "unconfirmed" && this.snap.state !== "failed") return;
    this.deps.log("critical", `${this.action}: ej bekräftat – meddelandet stängdes`);
    this.clearTimers();
    this.generation += 1;
    this.pending.clear();
    this.set({ state: "idle", text: null, onboardReceived: false, clientActionId: null });
  }

  /** ActionOutcome from WS `vehicle.action` or from polling. */
  outcome(o: ActionOutcome): void {
    const entry = this.pending.get(o.action_id);
    if (!entry) return; // older press or another client's action
    const st = this.snap.state;
    if (st !== "sent" && st !== "unconfirmed") return;
    if (o.paths.sdlink === "sent") this.set({ ...this.snap, onboardReceived: true });
    if (FINAL_OK.has(o.result)) {
      entry.acked = true;
      if ([...this.pending.values()].every((e) => e.acked)) this.toAcked();
    } else if (o.result === "failed") {
      this.toFailed(
        o.detail_sv
          ? { key: "detail", detail: o.detail_sv }
          : { key: "safety.error.vehicleFailed" },
      );
    }
  }

  // --- internals -------------------------------------------------------------------

  private url(): string {
    if (this.target !== null) return paths.vehicleAction(this.target, this.action);
    // trigger() never starts a Resume without a vehicle, so this is stop or return_home.
    return paths.allVehiclesAction(this.action as "stop" | "return_home");
  }

  private async attempt(gen: number, clientActionId: string): Promise<void> {
    const ctrl = new AbortController();
    const abort = this.after(TIMING.attemptTimeoutMs, () => ctrl.abort());
    // The 3 s limit covers headers AND body: a stalled body is retried like a timeout
    // (same client_action_id, so the server treats it as the same press).
    let res: Response;
    let body: unknown = null;
    try {
      res = await this.deps.fetch(this.url(), {
        method: "POST",
        credentials: "same-origin",
        redirect: "manual",
        headers: { ...MUTATION_HEADERS },
        body: JSON.stringify({ client_action_id: clientActionId }),
        signal: ctrl.signal,
      });
      if (res.status === 202 || res.status === 200) body = await res.json();
      else if (res.status !== 502 && res.status !== 504) body = await readError(res);
    } catch {
      this.cancel(abort);
      if (gen === this.generation) this.retryOrGiveUp(gen, clientActionId);
      return;
    }
    this.cancel(abort);
    if (gen !== this.generation) return;

    if (res.status === 202 || res.status === 200) {
      this.toSent(gen, body as ActionDispatched | ActionDispatchedAll | null);
      return;
    }
    if (res.status === 502 || res.status === 504) {
      this.retryOrGiveUp(gen, clientActionId);
      return;
    }
    const err = body as Partial<ApiError> | null;
    if (res.status === 401) {
      this.deps.onUnauthorized?.();
      this.toFailed({ key: "safety.error.session" });
    } else if (res.status === 403) {
      this.toFailed({ key: "safety.error.forbidden" });
    } else if (res.status === 503 && err?.code === "NO_PATH") {
      this.toFailed({ key: "safety.error.noPath" });
    } else if (err?.detail_sv) {
      this.toFailed({ key: "detail", detail: err.detail_sv });
    } else {
      this.toFailed({ key: "safety.state.failed" });
    }
  }

  private retryOrGiveUp(gen: number, clientActionId: string): void {
    if (this.now() - this.pressedAt >= TIMING.giveUpMs) {
      this.toFailed({ key: "safety.error.noGround" });
      return;
    }
    this.set({ ...this.snap, state: "retrying", text: { key: "safety.state.retrying" } });
    this.deps.log("info", `${this.action}: inget svar, försöker igen (${clientActionId})`);
    this.after(TIMING.retryDelayMs, () => {
      if (gen === this.generation) void this.attempt(gen, clientActionId);
    });
  }

  private toSent(gen: number, body: ActionDispatched | ActionDispatchedAll | null): void {
    this.pending.clear();
    if (body && "actions" in body) {
      for (const a of body.actions)
        this.pending.set(a.action_id, { vehicleId: a.vehicle_id, acked: false });
    } else if (body && "action_id" in body) {
      this.pending.set(body.action_id, { vehicleId: this.target ?? "", acked: false });
    }
    this.set({ ...this.snap, state: "sent", text: { key: "safety.state.sent" } });
    this.deps.log("info", `${this.action}: skickat`);
    this.after(TIMING.ackWaitMs, () => {
      if (gen === this.generation && this.snap.state === "sent") {
        this.set({ ...this.snap, state: "unconfirmed", text: { key: "safety.state.unconfirmed" } });
        this.armAlarm(gen);
      }
    });
    // Poll for the outcome in case the WebSocket is down or slow.
    const pollStart = this.now();
    const poll = (): void => {
      if (gen !== this.generation) return;
      if (this.snap.state !== "sent" && this.snap.state !== "unconfirmed") return;
      if (this.now() - pollStart > TIMING.pollForMs) return;
      for (const [actionId, e] of this.pending) {
        if (!e.acked) void this.pollOne(gen, e.vehicleId, actionId);
      }
      this.after(TIMING.pollEveryMs, poll);
    };
    this.after(TIMING.pollAfterMs, poll);
  }

  private async pollOne(gen: number, vehicleId: string, actionId: string): Promise<void> {
    try {
      const res = await this.deps.fetch(paths.vehicleActionOutcome(vehicleId, actionId), {
        credentials: "same-origin",
        redirect: "manual",
        headers: { Accept: "application/json" },
      });
      if (!res.ok || gen !== this.generation) return;
      this.outcome((await res.json()) as ActionOutcome);
    } catch {
      // the next poll tries again
    }
  }

  private toAcked(): void {
    this.clearTimers();
    const gen = this.generation;
    this.set({ ...this.snap, state: "acked", text: { key: "safety.state.acked" } });
    this.deps.log("info", `${this.action}: kvitterat`);
    this.after(TIMING.ackedShownMs, () => {
      if (gen === this.generation && this.snap.state === "acked") {
        this.set({ state: "idle", text: null, onboardReceived: false, clientActionId: null });
      }
    });
  }

  private toFailed(text: ActionText): void {
    this.clearTimers();
    this.set({ ...this.snap, state: "failed", text });
    this.deps.log("info", `${this.action}: misslyckades`);
    this.armAlarm(this.generation);
  }

  private armAlarm(gen: number): void {
    this.after(TIMING.alarmAfterMs, () => {
      const st = this.snap.state;
      if (gen === this.generation && (st === "unconfirmed" || st === "failed")) {
        this.deps.log("critical", `${this.action}: ej bekräftat efter 30 s`);
      }
    });
  }

  private after(ms: number, fn: () => void): Timer {
    const t = setTimeout(() => {
      this.timers.delete(t);
      fn();
    }, ms);
    this.timers.add(t);
    return t;
  }

  private cancel(t: Timer): void {
    clearTimeout(t);
    this.timers.delete(t);
  }

  private clearTimers(): void {
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
  }

  private set(s: ActionSnapshot): void {
    this.snap = s;
    for (const fn of this.listeners) fn(s);
  }
}

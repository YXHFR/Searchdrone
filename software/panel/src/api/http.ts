import type { ApiError } from "./types";

export const DEFAULT_TIMEOUT_MS = 10_000;

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly body: Partial<ApiError> | null,
  ) {
    super(body?.detail_sv ?? `HTTP ${status}`);
  }
  get code(): string | undefined {
    return this.body?.code;
  }
}

type Listener = (status: number) => void;
const unauthorizedListeners = new Set<Listener>();

/** Called on every 401 from a normal API call (not from /session itself). */
export function onUnauthorized(fn: Listener): () => void {
  unauthorizedListeners.add(fn);
  return () => unauthorizedListeners.delete(fn);
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  timeoutMs?: number;
  /** Do not report 401 to the re-login sheet (used by the login flow itself). */
  quiet401?: boolean;
}

/** Headers every changing request carries (CSRF guard on the server). Never Idempotency-Key. */
export const MUTATION_HEADERS: Readonly<Record<string, string>> = {
  "X-SD-Client": "panel",
  "Content-Type": "application/json",
};

export async function readError(res: Response): Promise<Partial<ApiError> | null> {
  try {
    return (await res.json()) as Partial<ApiError>;
  } catch {
    return null;
  }
}

/** Same-origin fetch: central headers, timeout, no redirects, problem-JSON errors. */
export async function api<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const method = opts.method ?? "GET";
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  try {
    const init: RequestInit = {
      method,
      credentials: "same-origin",
      redirect: "manual",
      signal: ctrl.signal,
      headers: method === "GET" ? { Accept: "application/json" } : { ...MUTATION_HEADERS },
    };
    if (method !== "GET") init.body = JSON.stringify(opts.body ?? {});
    const res = await fetch(path, init);
    if (!res.ok) {
      if (res.status === 401 && !opts.quiet401) {
        for (const fn of unauthorizedListeners) fn(401);
      }
      throw new HttpError(res.status, await readError(res));
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

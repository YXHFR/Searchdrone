// The files the SITL harness (simulation/tests/ground/test_panel_sitl.py) and the browser
// test share in build/panel-sitl/.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

export const EXCHANGE =
  process.env.SD_PANEL_SITL_EXCHANGE ??
  resolve(import.meta.dirname, "../../../../build/panel-sitl");

export interface Stack {
  url: string;
  login_code: string;
  /** The LAN listener (HTTPS, self-signed) and a pairing PIN, as `sd-ground pair` gives. */
  lan_url: string;
  pin: string;
}

/** Waits for the harness to have the stack up and returns its URL and login code. */
export async function waitForStack(limitMs = 600_000): Promise<Stack> {
  const path = join(EXCHANGE, "stack.json");
  const until = Date.now() + limitMs;
  while (Date.now() < until) {
    if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8")) as Stack;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`${path} kom aldrig: startade SITL-stacken?`);
}

export function readStack(): Stack {
  return JSON.parse(readFileSync(join(EXCHANGE, "stack.json"), "utf8")) as Stack;
}

export function stackUrl(): string {
  return readStack().url;
}

/** Asks the harness to stop sd-brain (the AI process) now. */
export function stopBrain(): void {
  writeFileSync(join(EXCHANGE, "stop-brain"), "stop\n");
}

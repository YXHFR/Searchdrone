// UUIDv7 (RFC 9562 §5.7): 48-bit Unix milliseconds, version 7, 74 random bits.
// Used for every client_*_id the panel sends. Monotonic within one page.
let lastMs = 0;
let lastRand = new Uint8Array(10);

function increment(bytes: Uint8Array): void {
  for (let i = bytes.length - 1; i >= 0; i--) {
    const next = (bytes[i] ?? 0) + 1;
    bytes[i] = next & 0xff;
    if (next <= 0xff) return;
  }
}

export function uuidv7(now: number = Date.now()): string {
  let ms = Math.max(now, lastMs);
  const rand = new Uint8Array(10);
  if (ms === lastMs) {
    rand.set(lastRand);
    increment(rand);
    // Counter overflow: borrow the next millisecond so order is kept.
    if (rand.every((b) => b === 0)) ms += 1;
  } else {
    crypto.getRandomValues(rand);
  }
  lastMs = ms;
  lastRand = rand;

  const b = new Uint8Array(16);
  for (let i = 0; i < 6; i++) b[i] = Math.floor(ms / 2 ** (8 * (5 - i))) & 0xff;
  b.set(rand, 6);
  b[6] = 0x70 | ((b[6] ?? 0) & 0x0f);
  b[8] = 0x80 | ((b[8] ?? 0) & 0x3f);
  const hex = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export const UUIDV7_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

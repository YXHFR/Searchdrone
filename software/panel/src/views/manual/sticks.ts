// Pure helpers for manual control. Axis values are in [-1, 1]; throttle 0 = hold altitude
// (the flight controller holds height and position in LOITER when the sticks are released).
export const MANUAL_RATE_HZ = 15;
export const DEADZONE = 0.08;

export interface Axes {
  roll: number;
  pitch: number;
  yaw: number;
  throttle: number;
}

export const NEUTRAL: Axes = { roll: 0, pitch: 0, yaw: 0, throttle: 0 };

export function clamp1(x: number): number {
  if (!Number.isFinite(x) || x === 0) return 0; // also turns -0 into 0
  return Math.max(-1, Math.min(1, x));
}

/** Values inside the deadzone are 0; outside it the range is rescaled to keep full travel. */
export function deadzone(x: number, dz: number = DEADZONE): number {
  const c = clamp1(x);
  const a = Math.abs(c);
  if (a <= dz) return 0;
  return Math.sign(c) * ((a - dz) / (1 - dz));
}

/** Pointer offset from the stick centre (screen pixels, y down) → stick position, clamped to the circle. */
export function stickFromPointer(dx: number, dy: number, radius: number): { x: number; y: number } {
  if (radius <= 0) return { x: 0, y: 0 };
  let x = dx / radius;
  let y = -dy / radius;
  const len = Math.hypot(x, y);
  if (len > 1) {
    x /= len;
    y /= len;
  }
  return { x: deadzone(x), y: deadzone(y) };
}

/** Mode 2 layout: left stick = yaw (x) and throttle (y), right stick = roll (x) and pitch (y). */
export function axesFromSticks(
  left: { x: number; y: number },
  right: { x: number; y: number },
): Axes {
  return {
    yaw: clamp1(left.x),
    throttle: clamp1(left.y),
    roll: clamp1(right.x),
    pitch: clamp1(right.y),
  };
}

/** Standard gamepad mapping: axes 0/1 left stick, 2/3 right stick, y down. */
export function axesFromGamepad(axes: readonly number[]): Axes {
  const a = (i: number): number => deadzone(axes[i] ?? 0);
  return axesFromSticks({ x: a(0), y: -a(1) }, { x: a(2), y: -a(3) });
}

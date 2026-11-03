// One short tone, for the moment a closer look is done and waits for the pilot. Silent where
// the browser has no Web Audio or has not let the page make sound yet (no click so far).
let ctx: AudioContext | null = null;

export function chime(): void {
  try {
    const Ctor = globalThis.AudioContext;
    if (typeof Ctor !== "function") return;
    ctx = ctx ?? new Ctor();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.45);
  } catch {
    // no sound is never an error
  }
}
